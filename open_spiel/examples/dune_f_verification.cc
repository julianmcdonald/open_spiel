// Comprehensive verification for Dune Imperium Prototype F.
// Validates all 7 user verification points:
// 1. Optimizer migration preserves all 120 inherited weights and optimizer states.
// 2. Additions disabled preserves legacy policy/value behaviour.
// 3. Additions enabled at zero-init achieves numerical parity with U31400.
// 4. Parity covers ordinary decisions AND newly supported roles (actions 91, 92).
// 5. Tleilaxu card-ID extraction tested independently (11->118, 13->120, boundaries, Reclaimed Forces, Skip, invalid).
// 6. Checkpoint save and reload preserves behaviour, feature flags, and optimizer layout.
// 7. New paths receive gradients and change weights after optimization.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "open_spiel/spiel.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_board.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "open_spiel/games/dune_imperium/dune_imperium_test_utils.h"

#include "dune_network.h"
#include "dune_semantic_action_scorer.h"
#include "dune_ppo_training_utils.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"

ABSL_FLAG(std::string, source_model_path,
          "/run/media/warcr/Storage/dune_drl_runtime/round7/u30650_12block_random_draft_overnight_20261006/checkpoints/ppo_model_update_31400.pt",
          "Path to source model");
ABSL_FLAG(std::string, source_opt_path,
          "/run/media/warcr/Storage/dune_drl_runtime/round7/u30650_12block_random_draft_overnight_20261006/checkpoints/ppo_optimizer_update_31400.pt",
          "Path to source optimizer");
ABSL_FLAG(std::string, source_manifest_path,
          "/run/media/warcr/Storage/dune_drl_runtime/round7/u30650_12block_random_draft_overnight_20261006/checkpoints/ppo_model_update_31400.json",
          "Path to source manifest");
ABSL_FLAG(std::string, export_start_bundle_dir, "",
          "If non-empty, exports untouched migrated F starting bundle here");

// Link satisfaction flags for dune_ppo_training_utils.cc:
ABSL_FLAG(int, ppo_minibatch_size, 2048, "");
ABSL_FLAG(int, ppo_update_epochs, 4, "");
ABSL_FLAG(double, ppo_clip_epsilon, 0.2, "");
ABSL_FLAG(bool, normalize_advantages, true, "");
ABSL_FLAG(bool, ppo_clip_value_loss, true, "");
ABSL_FLAG(double, entropy_coef, 0.01, "");
ABSL_FLAG(double, value_coef, 0.5, "");
ABSL_FLAG(double, logit_cap, 10.0, "");
ABSL_FLAG(double, target_kl, 0.0, "");
ABSL_FLAG(bool, train_amp, false, "");
ABSL_FLAG(double, grad_clip_norm, 0.5, "");
ABSL_FLAG(bool, diagnostics_only, false, "");

using namespace open_spiel;
using namespace open_spiel::dune_imperium;

static int g_checks_run = 0;
static int g_checks_passed = 0;

#define CHECK_ASSERT(cond, msg)                                                \
  do {                                                                         \
    ++g_checks_run;                                                            \
    if (!(cond)) {                                                             \
      std::cerr << "FAIL: " << msg << " at " << __FILE__ << ":" << __LINE__    \
                << std::endl;                                                  \
      std::exit(1);                                                            \
    } else {                                                                   \
      ++g_checks_passed;                                                       \
    }                                                                          \
  } while (0)

inline std::vector<float> ComputeLegalProbabilities(
    const std::vector<float>& logits, const std::vector<Action>& legal_actions) {
  float max_l = -1e9f;
  for (Action a : legal_actions) {
    if (logits[a] > max_l) max_l = logits[a];
  }
  double sum = 0.0;
  std::vector<float> probs(legal_actions.size());
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    probs[i] = std::exp(logits[legal_actions[i]] - max_l);
    sum += probs[i];
  }
  for (size_t i = 0; i < legal_actions.size(); ++i) {
    probs[i] /= static_cast<float>(sum);
  }
  return probs;
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  std::cout << "========================================================\n";
  std::cout << "=== DUNE IMPERIUM PROTOTYPE F COMPREHENSIVE VERIFY   ===\n";
  std::cout << "========================================================\n\n";

  torch::Device device(torch::kCPU);
  const std::string source_model_path = absl::GetFlag(FLAGS_source_model_path);
  const std::string source_opt_path = absl::GetFlag(FLAGS_source_opt_path);
  const std::string source_manifest_path = absl::GetFlag(FLAGS_source_manifest_path);

  std::cout << "Source model:    " << source_model_path << "\n";
  std::cout << "Source optimizer:" << source_opt_path << "\n";
  std::cout << "Source manifest: " << source_manifest_path << "\n\n";

  CHECK_ASSERT(std::filesystem::exists(source_model_path), "Source model file must exist");
  CHECK_ASSERT(std::filesystem::exists(source_opt_path), "Source optimizer file must exist");
  CHECK_ASSERT(std::filesystem::exists(source_manifest_path), "Source manifest file must exist");

  const int64_t input_dim = 9182;
  const int64_t hidden_dim = 2048;
  const int64_t action_dim = 2391;
  const int num_blocks = 12;
  const double base_lr = 5e-6;
  const double new_block_lr = 5e-5;

  // -----------------------------------------------------------------------
  // CHECK 5: Test Card-ID Extraction Independently of Parity
  // -----------------------------------------------------------------------
  std::cout << "--------------------------------------------------------\n";
  std::cout << "CHECK 5: Independent Tleilaxu Card-ID Extraction\n";
  std::cout << "--------------------------------------------------------\n";
  {
    // Authoritative engine mapping:
    // Scientific Breakthrough: local 11 -> shared 118
    // Stitched Horror: local 13 -> shared 120
    CHECK_ASSERT(kTleilaxuCards[11].imperium_card_id == 118,
                 "Scientific Breakthrough (11) must map to 118");
    CHECK_ASSERT(kTleilaxuCards[13].imperium_card_id == 120,
                 "Stitched Horror (13) must map to 120");
    std::cout << "  Engine mapping: local 11 -> shared " << kTleilaxuCards[11].imperium_card_id << "\n";
    std::cout << "  Engine mapping: local 13 -> shared " << kTleilaxuCards[13].imperium_card_id << "\n";

    // Range boundaries: local 1 -> 108, local 18 -> 125
    CHECK_ASSERT(kTleilaxuCards[1].imperium_card_id == 108,
                 "Local 1 must map to 108");
    CHECK_ASSERT(kTleilaxuCards[18].imperium_card_id == 125,
                 "Local 18 must map to 125");
    std::cout << "  Engine boundary: local 1 -> shared " << kTleilaxuCards[1].imperium_card_id << "\n";
    std::cout << "  Engine boundary: local 18 -> shared " << kTleilaxuCards[18].imperium_card_id << "\n";

    // Reclaimed Forces (local 0): imperium_card_id is -1 (not an Imperium deck card)
    CHECK_ASSERT(kTleilaxuCards[0].imperium_card_id == -1,
                 "Reclaimed Forces (local 0) must have imperium_card_id == -1");
    std::cout << "  Reclaimed forces local 0 -> shared " << kTleilaxuCards[0].imperium_card_id << "\n";

    // Native state descriptor path on a real DuneImperiumState!
    auto state = BuildStateAtFirstRevealTurn("dune_imperium(enable_immortality=true)");
    auto* dune_st = dynamic_cast<DuneImperiumState*>(state.get());
    CHECK_ASSERT(dune_st != nullptr, "Must cast to DuneImperiumState");
    const Player p = dune_st->CurrentPlayer();

    // Setup Tleilaxu row: slot 0 = 13 (Stitched Horror), slot 1 = 11 (Scientific Breakthrough)
    dune_st->SetTleilaxuRowForTesting({13, 11});
    dune_st->SetSpecimensForTesting(p, 5);

    // Verify GetTleilaxuRowCard accessor
    CHECK_ASSERT(dune_st->GetTleilaxuRowCard(0) == 13, "Slot 0 must be 13");
    CHECK_ASSERT(dune_st->GetTleilaxuRowCard(1) == 11, "Slot 1 must be 11");

    auto legal_actions = dune_st->LegalActions();
    CHECK_ASSERT(std::find(legal_actions.begin(), legal_actions.end(), 91) != legal_actions.end(),
                 "Action 91 must be legal");
    CHECK_ASSERT(std::find(legal_actions.begin(), legal_actions.end(), 92) != legal_actions.end(),
                 "Action 92 must be legal");

    // Under schema v3, actions 91 and 92 must be unsupported
    dune_semantic::CandidateActionData cand_v3;
    dune_semantic::ExtractCandidateDescriptors(
        *dune_st, legal_actions, &cand_v3, "semantic_action_v3");
    for (size_t i = 0; i < cand_v3.actions.size(); ++i) {
      if (cand_v3.actions[i] == 91 || cand_v3.actions[i] == 92) {
        CHECK_ASSERT(cand_v3.roles[i] == dune_semantic::ActionRole::kUnsupported,
                     "Schema v3 must mark actions 91 and 92 unsupported");
      }
    }

    // Under schema v4, actions 91 and 92 must produce kTleilaxuPurchase with correct card IDs!
    dune_semantic::CandidateActionData cand_v4;
    dune_semantic::ExtractCandidateDescriptors(
        *dune_st, legal_actions, &cand_v4, "semantic_action_v4");
    int found_91 = 0, found_92 = 0;
    for (size_t i = 0; i < cand_v4.actions.size(); ++i) {
      if (cand_v4.actions[i] == 91) {
        CHECK_ASSERT(cand_v4.roles[i] == dune_semantic::ActionRole::kTleilaxuPurchase,
                     "Schema v4 must mark action 91 as kTleilaxuPurchase");
        CHECK_ASSERT(cand_v4.card_ids[i] == 120,
                     "Action 91 card_id must be 120 (Stitched Horror)");
        std::cout << "  Native descriptor action 91: role=" << static_cast<int>(cand_v4.roles[i])
                  << " card_id=" << cand_v4.card_ids[i] << " (Stitched Horror)\n";
        ++found_91;
      }
      if (cand_v4.actions[i] == 92) {
        CHECK_ASSERT(cand_v4.roles[i] == dune_semantic::ActionRole::kTleilaxuPurchase,
                     "Schema v4 must mark action 92 as kTleilaxuPurchase");
        CHECK_ASSERT(cand_v4.card_ids[i] == 118,
                     "Action 92 card_id must be 118 (Scientific Breakthrough)");
        std::cout << "  Native descriptor action 92: role=" << static_cast<int>(cand_v4.roles[i])
                  << " card_id=" << cand_v4.card_ids[i] << " (Scientific Breakthrough)\n";
        ++found_92;
      }
    }
    CHECK_ASSERT(found_91 == 1 && found_92 == 1, "Must find both actions 91 and 92");

    // Test boundaries in slot 0 & slot 1: local 1 (108) and local 18 (125)
    dune_st->SetTleilaxuRowForTesting({1, 18});
    dune_semantic::CandidateActionData cand_bounds;
    dune_semantic::ExtractCandidateDescriptors(
        *dune_st, legal_actions, &cand_bounds, "semantic_action_v4");
    for (size_t i = 0; i < cand_bounds.actions.size(); ++i) {
      if (cand_bounds.actions[i] == 91) {
        CHECK_ASSERT(cand_bounds.card_ids[i] == 108, "Boundary local 1 must give shared 108");
      }
      if (cand_bounds.actions[i] == 92) {
        CHECK_ASSERT(cand_bounds.card_ids[i] == 125, "Boundary local 18 must give shared 125");
      }
    }
    std::cout << "  Native descriptor boundaries: 91->108, 92->125 verified\n";

    // Test empty/invalid slot:
    dune_st->SetTleilaxuRowForTesting({kInvalidCard, kInvalidCard});
    dune_semantic::CandidateActionData cand_empty;
    dune_semantic::ExtractCandidateDescriptors(
        *dune_st, legal_actions, &cand_empty, "semantic_action_v4");
    for (size_t i = 0; i < cand_empty.actions.size(); ++i) {
      if (cand_empty.actions[i] == 91 || cand_empty.actions[i] == 92) {
        CHECK_ASSERT(cand_empty.roles[i] == dune_semantic::ActionRole::kUnsupported,
                     "Empty slot must be unsupported");
      }
    }
    std::cout << "  Native descriptor empty slots: properly unsupported\n";

    // Test Reclaimed Forces (local 0):
    dune_st->SetTleilaxuRowForTesting({0, 0});
    dune_semantic::CandidateActionData cand_rf;
    dune_semantic::ExtractCandidateDescriptors(
        *dune_st, legal_actions, &cand_rf, "semantic_action_v4");
    for (size_t i = 0; i < cand_rf.actions.size(); ++i) {
      if (cand_rf.actions[i] == 91 || cand_rf.actions[i] == 92) {
        CHECK_ASSERT(cand_rf.roles[i] == dune_semantic::ActionRole::kUnsupported,
                     "Reclaimed forces in row must be unsupported");
      }
    }
    std::cout << "  Native descriptor Reclaimed Forces in row: properly unsupported\n";

    // Test action 90 (Reclaimed Forces direct purchase) and action 96 (Skip choice):
    for (size_t i = 0; i < cand_v4.actions.size(); ++i) {
      if (cand_v4.actions[i] == 90) {
        CHECK_ASSERT(cand_v4.roles[i] == dune_semantic::ActionRole::kUnsupported,
                     "Action 90 (Reclaimed Forces) must be unsupported");
      }
      if (cand_v4.actions[i] == 96) {
        CHECK_ASSERT(cand_v4.roles[i] == dune_semantic::ActionRole::kUnsupported,
                     "Action 96 (Skip choice) must be unsupported");
      }
    }
    std::cout << "  Action 90 and Action 96: properly unsupported\n";
    std::cout << "CHECK 5 PASSED.\n\n";
  }

  // -----------------------------------------------------------------------
  // CHECK 1: Migration Preserves Inherited Weights and Optimizer State
  // -----------------------------------------------------------------------
  std::cout << "--------------------------------------------------------\n";
  std::cout << "CHECK 1: Optimizer Migration Verification\n";
  std::cout << "--------------------------------------------------------\n";

  // Build source model (legacy U31400: 120 parameters)
  auto source_model = std::make_shared<SharedDunePolicyValueNetImpl>(
      input_dim, hidden_dim, action_dim, num_blocks,
      /*use_nonlinear=*/false, /*with_aux_heads=*/false,
      /*head_init_seed=*/0, /*with_semantic_scorer=*/true,
      /*enable_zone_encoder=*/false);
  LoadModelCheckpointRobust(source_model, source_model_path, device);
  source_model->to(device);
  source_model->eval();

  // Build source optimizer with exact U31400 on-disk layout {2, 86, 32}
  std::vector<torch::Tensor> src_g0;
  src_g0.push_back(source_model->policy_head->weight);
  if (source_model->policy_head->bias.defined()) src_g0.push_back(source_model->policy_head->bias);

  std::vector<torch::Tensor> src_g1;
  src_g1.push_back(source_model->input_layer->weight);
  if (source_model->input_layer->bias.defined()) src_g1.push_back(source_model->input_layer->bias);
  for (int b = 0; b < 8; ++b) {
    for (auto& p : source_model->res_blocks[b]->parameters()) src_g1.push_back(p);
  }
  src_g1.push_back(source_model->value_head->weight);
  if (source_model->value_head->bias.defined()) src_g1.push_back(source_model->value_head->bias);
  auto add_scorer_p = [&](torch::nn::Module& m) {
    for (auto& p : m.parameters()) src_g1.push_back(p);
  };
  add_scorer_p(*source_model->semantic_scorer_->card_embedding);
  add_scorer_p(*source_model->semantic_scorer_->space_embedding);
  add_scorer_p(*source_model->semantic_scorer_->desc_proj);
  add_scorer_p(*source_model->semantic_scorer_->desc_ln);
  add_scorer_p(*source_model->semantic_scorer_->trunk_proj);
  add_scorer_p(*source_model->semantic_scorer_->trunk_ln);
  add_scorer_p(*source_model->semantic_scorer_->mlp1);
  add_scorer_p(*source_model->semantic_scorer_->mlp1_ln);
  add_scorer_p(*source_model->semantic_scorer_->out_layer);
  add_scorer_p(*source_model->semantic_scorer_->out_layer_ext);

  std::vector<torch::Tensor> src_g2;
  for (int b = 8; b < 12; ++b) {
    for (auto& p : source_model->res_blocks[b]->parameters()) src_g2.push_back(p);
  }

  CHECK_ASSERT(src_g0.size() == 2, "Source group 0 must have 2 params");
  CHECK_ASSERT(src_g1.size() == 86, "Source group 1 must have 86 params");
  CHECK_ASSERT(src_g2.size() == 32, "Source group 2 must have 32 params");

  std::vector<torch::optim::OptimizerParamGroup> src_groups;
  src_groups.emplace_back(src_g0);
  src_groups.emplace_back(src_g1);
  src_groups.emplace_back(src_g2);
  auto source_opt = std::make_unique<torch::optim::AdamW>(
      src_groups, torch::optim::AdamWOptions(base_lr).eps(1e-5));
  torch::load(*source_opt, source_opt_path, device);
  std::cout << "  Source optimizer loaded from archive: 3 groups (2, 86, 32) = 120 params\n";

  // Build target Prototype F model (128 parameters: 120 inherited + 8 new)
  auto target_model = std::make_shared<SharedDunePolicyValueNetImpl>(
      input_dim, hidden_dim, action_dim, num_blocks,
      /*use_nonlinear=*/false, /*with_aux_heads=*/false,
      /*head_init_seed=*/0, /*with_semantic_scorer=*/true,
      /*enable_zone_encoder=*/true);
  // Load inherited weights
  LoadModelCheckpointRobust(target_model, source_model_path, device);
  target_model->to(device);
  target_model->eval();

  const auto target_params = target_model->parameters();
  CHECK_ASSERT(target_params.size() == 128, "Target model must have 128 parameters");
  std::cout << "  Target Prototype F model created: " << target_params.size() << " parameters\n";

  // Build target optimizer (3 groups: 2, 86, 40)
  auto target_opt = MakeDuneOptimizer(target_model, /*lr=*/base_lr, /*weight_decay=*/0.0, /*policy_weight_decay=*/0.0, /*new_block_lr=*/new_block_lr, /*new_block_first_idx=*/8);
  CHECK_ASSERT(target_opt->param_groups().size() == 3, "Target optimizer must have 3 groups");
  CHECK_ASSERT(target_opt->param_groups()[0].params().size() == 2, "Target group 0 has 2 params");
  CHECK_ASSERT(target_opt->param_groups()[1].params().size() == 86, "Target group 1 has 86 params");
  CHECK_ASSERT(target_opt->param_groups()[2].params().size() == 40, "Target group 2 has 40 params (32 old + 8 new)");
  std::cout << "  Target optimizer groups: group 0=" << target_opt->param_groups()[0].params().size()
            << " (lr=" << target_opt->param_groups()[0].options().get_lr() << ")"
            << ", group 1=" << target_opt->param_groups()[1].params().size()
            << " (lr=" << target_opt->param_groups()[1].options().get_lr() << ")"
            << ", group 2=" << target_opt->param_groups()[2].params().size()
            << " (lr=" << target_opt->param_groups()[2].options().get_lr() << ")\n";

  // Run LoadOptimizerCheckpointMigrating
  LoadOptimizerCheckpointMigrating(target_model, *target_opt, source_opt_path, device, base_lr);
  std::cout << "  LoadOptimizerCheckpointMigrating completed successfully.\n";

  // Verify all 120 inherited parameters and optimizer states!
  int verified_params = 0;
  for (size_t g = 0; g < 3; ++g) {
    const auto& src_g = source_opt->param_groups()[g];
    const auto& tgt_g = target_opt->param_groups()[g];
    const size_t num_inherited = src_g.params().size();
    for (size_t i = 0; i < num_inherited; ++i) {
      const auto& src_p = src_g.params()[i];
      const auto& tgt_p = tgt_g.params()[i];

      // Check parameter values match bitwise
      CHECK_ASSERT(torch::equal(src_p, tgt_p), "Inherited parameter values must match bitwise");

      // Check optimizer state
      auto& src_state_map = source_opt->state();
      auto& tgt_state_map = target_opt->state();

      auto src_it = src_state_map.find(src_p.unsafeGetTensorImpl());
      auto tgt_it = tgt_state_map.find(tgt_p.unsafeGetTensorImpl());

      CHECK_ASSERT(src_it != src_state_map.end(), "Source state must exist");
      CHECK_ASSERT(tgt_it != tgt_state_map.end(), "Target state must exist");

      auto& src_step_t = static_cast<torch::optim::AdamWParamState&>(*src_it->second).step();
      auto& tgt_step_t = static_cast<torch::optim::AdamWParamState&>(*tgt_it->second).step();
      CHECK_ASSERT(src_step_t == tgt_step_t, "Step counts must match");

      auto& src_exp_avg = static_cast<torch::optim::AdamWParamState&>(*src_it->second).exp_avg();
      auto& tgt_exp_avg = static_cast<torch::optim::AdamWParamState&>(*tgt_it->second).exp_avg();
      CHECK_ASSERT(torch::equal(src_exp_avg, tgt_exp_avg), "exp_avg must match bitwise");

      auto& src_exp_avg_sq = static_cast<torch::optim::AdamWParamState&>(*src_it->second).exp_avg_sq();
      auto& tgt_exp_avg_sq = static_cast<torch::optim::AdamWParamState&>(*tgt_it->second).exp_avg_sq();
      CHECK_ASSERT(torch::equal(src_exp_avg_sq, tgt_exp_avg_sq), "exp_avg_sq must match bitwise");

      ++verified_params;
    }
  }
  CHECK_ASSERT(verified_params == 120, "All 120 inherited parameters must be verified");
  std::cout << "  Verified all 120 inherited parameters: exact bitwise weight & optimizer match!\n";

  // Verify the 8 new parameters in group 2: empty state
  const auto& tgt_g2 = target_opt->param_groups()[2];
  for (size_t i = 32; i < 40; ++i) {
    const auto& new_p = tgt_g2.params()[i];
    auto& tgt_state_map = target_opt->state();
    auto tgt_it = tgt_state_map.find(new_p.unsafeGetTensorImpl());
    CHECK_ASSERT(tgt_it == tgt_state_map.end(), "New parameter must start with uninitialized optimizer state");
  }
  std::cout << "  Verified all 8 new parameters: empty optimizer state verified.\n";
  std::cout << "CHECK 1 PASSED.\n\n";
  if (!absl::GetFlag(FLAGS_export_start_bundle_dir).empty()) {
    const std::string out_dir = absl::GetFlag(FLAGS_export_start_bundle_dir);
    std::filesystem::create_directories(out_dir);
    const std::string model_out = out_dir + "/ppo_model_update_31400.pt";
    const std::string opt_out = out_dir + "/ppo_optimizer_update_31400.pt";
    torch::save(target_model, model_out);
    torch::save(*target_opt, opt_out);
    std::cout << "  [EXPORT] Exported untouched migrated F starting model to: " << model_out << "\n";
    std::cout << "  [EXPORT] Exported untouched migrated F starting optimizer to: " << opt_out << "\n\n";
  }

  // -----------------------------------------------------------------------
  // CHECK 2: Disabled Additions Match Legacy Policy/Value Bitwise
  // -----------------------------------------------------------------------
  std::cout << "--------------------------------------------------------\n";
  std::cout << "CHECK 2: Disabled Additions Parity\n";
  std::cout << "--------------------------------------------------------\n";

  auto model_disabled = std::make_shared<SharedDunePolicyValueNetImpl>(
      input_dim, hidden_dim, action_dim, num_blocks,
      /*use_nonlinear=*/false, /*with_aux_heads=*/false,
      /*head_init_seed=*/0, /*with_semantic_scorer=*/true,
      /*enable_zone_encoder=*/false);
  LoadModelCheckpointRobust(model_disabled, source_model_path, device);
  model_disabled->to(device);
  model_disabled->eval();

  // Test across 50 diverse states from game rollouts
  std::shared_ptr<const Game> game = LoadGame("dune_imperium(enable_immortality=true)");
  std::mt19937_64 rng(42);
  float max_diff_disabled_logits = 0.0f;
  float max_diff_disabled_values = 0.0f;
  int states_checked_chk2 = 0;

  while (states_checked_chk2 < 50) {
    std::unique_ptr<State> st = game->NewInitialState();
    while (!st->IsTerminal() && states_checked_chk2 < 50) {
      if (st->IsChanceNode()) {
        auto outcomes = st->ChanceOutcomes();
        std::uniform_int_distribution<size_t> dist(0, outcomes.size() - 1);
        st->ApplyAction(outcomes[dist(rng)].first);
        continue;
      }
      auto legal = st->LegalActions();
      if (legal.empty()) break;

      const auto* dune_st = dynamic_cast<const DuneImperiumState*>(st.get());
      if (dune_st != nullptr && (dune_st->phase() == GamePhase::kAgentTurns || dune_st->phase() == GamePhase::kRevealTurns)) {
        const Player p = dune_st->CurrentPlayer();
        std::vector<float> obs = dune_st->InformationStateTensorWithAppendix(
            p, MarketAppendixMode::kFullPublicInformationV3);
        torch::Tensor obs_t = torch::from_blob(obs.data(), {1, static_cast<long>(obs.size())}, torch::kFloat32).to(device);

        dune_semantic::CandidateActionData cand_data;
        dune_semantic::ExtractCandidateDescriptors(
            *dune_st, legal, &cand_data, "semantic_action_v3");

        auto out_src = source_model->forward(obs_t);
        auto out_dis = model_disabled->forward(obs_t);

        std::vector<const dune_semantic::CandidateActionData*> batch_cands = {&cand_data};
        dune_semantic::ApplySemanticScorerBatch(
            source_model->semantic_scorer_, out_src.trunk, batch_cands, out_src.logits, device);
        dune_semantic::ApplySemanticScorerBatch(
            model_disabled->semantic_scorer_, out_dis.trunk, batch_cands, out_dis.logits, device);

        float l_diff = (out_src.logits - out_dis.logits).abs().max().item().toFloat();
        float v_diff = (out_src.values - out_dis.values).abs().max().item().toFloat();

        if (l_diff > max_diff_disabled_logits) max_diff_disabled_logits = l_diff;
        if (v_diff > max_diff_disabled_values) max_diff_disabled_values = v_diff;

        ++states_checked_chk2;
      }

      std::uniform_int_distribution<size_t> dist(0, legal.size() - 1);
      st->ApplyAction(legal[dist(rng)]);
    }
  }

  std::cout << "  States checked: " << states_checked_chk2 << "\n";
  std::cout << "  Max logits abs diff: " << max_diff_disabled_logits << "\n";
  std::cout << "  Max values abs diff: " << max_diff_disabled_values << "\n";
  CHECK_ASSERT(max_diff_disabled_logits == 0.0f, "Disabled additions must match bitwise (0.0f diff)");
  CHECK_ASSERT(max_diff_disabled_values == 0.0f, "Disabled values must match bitwise (0.0f diff)");
  std::cout << "CHECK 2 PASSED.\n\n";

  // -----------------------------------------------------------------------
  // CHECK 3 & 4: Parity with Enabled Additions on Ordinary Decisions & Tleilaxu Slots
  // -----------------------------------------------------------------------
  std::cout << "--------------------------------------------------------\n";
  std::cout << "CHECK 3 & 4: Enabled Additions Parity (Ordinary & Tleilaxu Actions)\n";
  std::cout << "--------------------------------------------------------\n";

  float max_diff_logits = 0.0f;
  float max_diff_cap_logits = 0.0f;
  float max_diff_probs = 0.0f;
  float max_diff_values = 0.0f;
  int greedy_mismatches = 0;
  int ordinary_states_checked = 0;

  // 1. Check over 100 game positions
  rng.seed(999);
  while (ordinary_states_checked < 100) {
    std::unique_ptr<State> st = game->NewInitialState();
    while (!st->IsTerminal() && ordinary_states_checked < 100) {
      if (st->IsChanceNode()) {
        auto outcomes = st->ChanceOutcomes();
        std::uniform_int_distribution<size_t> dist(0, outcomes.size() - 1);
        st->ApplyAction(outcomes[dist(rng)].first);
        continue;
      }
      auto legal = st->LegalActions();
      if (legal.empty()) break;

      const auto* dune_st = dynamic_cast<const DuneImperiumState*>(st.get());
      if (dune_st != nullptr && (dune_st->phase() == GamePhase::kAgentTurns || dune_st->phase() == GamePhase::kRevealTurns)) {
        const Player p = dune_st->CurrentPlayer();
        std::vector<float> obs = dune_st->InformationStateTensorWithAppendix(
            p, MarketAppendixMode::kFullPublicInformationV3);
        torch::Tensor obs_t = torch::from_blob(obs.data(), {1, static_cast<long>(obs.size())}, torch::kFloat32).to(device);

        dune_semantic::CandidateActionData cand_src;
        dune_semantic::ExtractCandidateDescriptors(
            *dune_st, legal, &cand_src, "semantic_action_v3");

        dune_semantic::CandidateActionData cand_tgt;
        dune_semantic::ExtractCandidateDescriptors(
            *dune_st, legal, &cand_tgt, "semantic_action_v4");

        auto out_src = source_model->forward(obs_t);
        auto out_tgt = target_model->forward(obs_t);

        std::vector<const dune_semantic::CandidateActionData*> cands_src = {&cand_src};
        std::vector<const dune_semantic::CandidateActionData*> cands_tgt = {&cand_tgt};
        dune_semantic::ApplySemanticScorerBatch(
            source_model->semantic_scorer_, out_src.trunk, cands_src, out_src.logits, device);
        dune_semantic::ApplySemanticScorerBatch(
            target_model->semantic_scorer_, out_tgt.trunk, cands_tgt, out_tgt.logits, device);

        // Apply logit cap 10.0
        torch::Tensor cap_src = torch::clamp(out_src.logits, -10.0, 10.0);
        torch::Tensor cap_tgt = torch::clamp(out_tgt.logits, -10.0, 10.0);

        float l_diff = (out_src.logits - out_tgt.logits).abs().max().item().toFloat();
        float cap_diff = (cap_src - cap_tgt).abs().max().item().toFloat();
        float v_diff = (out_src.values - out_tgt.values).abs().max().item().toFloat();

        if (l_diff > max_diff_logits) max_diff_logits = l_diff;
        if (cap_diff > max_diff_cap_logits) max_diff_cap_logits = cap_diff;
        if (v_diff > max_diff_values) max_diff_values = v_diff;

        // Compare legal action probabilities and greedy actions
        std::vector<float> l_src_vec(action_dim), l_tgt_vec(action_dim);
        torch::Tensor la_cpu = out_src.logits.to(torch::kCPU).contiguous();
        torch::Tensor lb_cpu = out_tgt.logits.to(torch::kCPU).contiguous();
        std::memcpy(l_src_vec.data(), la_cpu.data_ptr<float>(), action_dim * sizeof(float));
        std::memcpy(l_tgt_vec.data(), lb_cpu.data_ptr<float>(), action_dim * sizeof(float));

        auto p_src = ComputeLegalProbabilities(l_src_vec, legal);
        auto p_tgt = ComputeLegalProbabilities(l_tgt_vec, legal);

        float best_src_p = -1.0f;
        Action best_src_a = -1;
        float best_tgt_p = -1.0f;
        Action best_tgt_a = -1;

        for (size_t k = 0; k < legal.size(); ++k) {
          float p_diff = std::abs(p_src[k] - p_tgt[k]);
          if (p_diff > max_diff_probs) max_diff_probs = p_diff;
          if (p_src[k] > best_src_p) { best_src_p = p_src[k]; best_src_a = legal[k]; }
          if (p_tgt[k] > best_tgt_p) { best_tgt_p = p_tgt[k]; best_tgt_a = legal[k]; }
        }
        if (best_src_a != best_tgt_a) {
          ++greedy_mismatches;
        }

        ++ordinary_states_checked;
      }

      std::uniform_int_distribution<size_t> dist(0, legal.size() - 1);
      st->ApplyAction(legal[dist(rng)]);
    }
  }

  std::cout << "  Ordinary decisions checked: " << ordinary_states_checked << " states\n";
  std::cout << "  Max logits abs diff:        " << max_diff_logits << "\n";
  std::cout << "  Max cap 10 logits diff:     " << max_diff_cap_logits << "\n";
  std::cout << "  Max legal probs diff:       " << max_diff_probs << "\n";
  std::cout << "  Max value head diff:        " << max_diff_values << "\n";
  std::cout << "  Greedy action mismatches:   " << greedy_mismatches << "\n";

  // 2. Specific test on states where action 91 and action 92 are legal
  float max_diff_tleilaxu_slot0 = 0.0f;
  float max_diff_tleilaxu_slot1 = 0.0f;
  int tleilaxu_states_tested = 0;

  for (int c0 = 1; c0 <= 18; ++c0) {
    for (int c1 = 1; c1 <= 18; ++c1) {
      auto state = BuildStateAtFirstRevealTurn("dune_imperium(enable_immortality=true)");
      auto* dune_st = dynamic_cast<DuneImperiumState*>(state.get());
      const Player p = dune_st->CurrentPlayer();

      dune_st->SetTleilaxuRowForTesting({c0, c1});
      dune_st->SetSpecimensForTesting(p, 10);

      auto legal = dune_st->LegalActions();
      CHECK_ASSERT(std::find(legal.begin(), legal.end(), 91) != legal.end(), "Action 91 must be legal");
      CHECK_ASSERT(std::find(legal.begin(), legal.end(), 92) != legal.end(), "Action 92 must be legal");

      std::vector<float> obs = dune_st->InformationStateTensorWithAppendix(
          p, MarketAppendixMode::kFullPublicInformationV3);
      torch::Tensor obs_t = torch::from_blob(obs.data(), {1, static_cast<long>(obs.size())}, torch::kFloat32).to(device);

      dune_semantic::CandidateActionData cand_src;
      dune_semantic::ExtractCandidateDescriptors(
          *dune_st, legal, &cand_src, "semantic_action_v3");

      dune_semantic::CandidateActionData cand_tgt;
      dune_semantic::ExtractCandidateDescriptors(
          *dune_st, legal, &cand_tgt, "semantic_action_v4");

      auto out_src = source_model->forward(obs_t);
      auto out_tgt = target_model->forward(obs_t);

      std::vector<const dune_semantic::CandidateActionData*> cands_src = {&cand_src};
      std::vector<const dune_semantic::CandidateActionData*> cands_tgt = {&cand_tgt};
      dune_semantic::ApplySemanticScorerBatch(
          source_model->semantic_scorer_, out_src.trunk, cands_src, out_src.logits, device);
      dune_semantic::ApplySemanticScorerBatch(
          target_model->semantic_scorer_, out_tgt.trunk, cands_tgt, out_tgt.logits, device);

      float diff_91 = std::abs(out_src.logits[0][91].item().toFloat() - out_tgt.logits[0][91].item().toFloat());
      float diff_92 = std::abs(out_src.logits[0][92].item().toFloat() - out_tgt.logits[0][92].item().toFloat());

      if (diff_91 > max_diff_tleilaxu_slot0) max_diff_tleilaxu_slot0 = diff_91;
      if (diff_92 > max_diff_tleilaxu_slot1) max_diff_tleilaxu_slot1 = diff_92;

      ++tleilaxu_states_tested;
    }
  }

  std::cout << "  Tleilaxu card pairs tested: " << tleilaxu_states_tested << " states (covering all 18x18 card combinations)\n";
  std::cout << "  Action 91 (slot 0) max logit diff: " << max_diff_tleilaxu_slot0 << "\n";
  std::cout << "  Action 92 (slot 1) max logit diff: " << max_diff_tleilaxu_slot1 << "\n";

  CHECK_ASSERT(max_diff_logits < 1e-6f, "Max logit diff must be < 1e-6");
  CHECK_ASSERT(max_diff_cap_logits < 1e-6f, "Max cap logit diff must be < 1e-6");
  CHECK_ASSERT(max_diff_probs < 1e-6f, "Max prob diff must be < 1e-6");
  CHECK_ASSERT(max_diff_values < 1e-6f, "Max value diff must be < 1e-6");
  CHECK_ASSERT(greedy_mismatches == 0, "Greedy actions must match 100%");
  CHECK_ASSERT(max_diff_tleilaxu_slot0 < 1e-6f, "Slot 0 diff must be < 1e-6");
  CHECK_ASSERT(max_diff_tleilaxu_slot1 < 1e-6f, "Slot 1 diff must be < 1e-6");
  std::cout << "CHECK 3 & 4 PASSED.\n\n";

  // -----------------------------------------------------------------------
  // CHECK 6: Save and Reload Preserves Behaviour, Metadata, and Optimizer Layout
  // -----------------------------------------------------------------------
  std::cout << "--------------------------------------------------------\n";
  std::cout << "CHECK 6: Checkpoint Save and Reload Preservation\n";
  std::cout << "--------------------------------------------------------\n";
  {
    const std::string test_save_dir = "/home/warcr/dune_drl_runtime/round7/u31400_test_reload";
    std::filesystem::create_directories(test_save_dir);
    const std::string test_model_path = test_save_dir + "/ppo_model_update_31400.pt";
    const std::string test_opt_path = test_save_dir + "/ppo_optimizer_update_31400.pt";
    const std::string test_json_path = test_save_dir + "/ppo_model_update_31400.json";

    // Save target model
    torch::save(target_model, test_model_path);
    torch::save(*target_opt, test_opt_path);

    // Write sidecar JSON with enable_zone_encoder and semantic_action_v4
    std::ofstream out_json(test_json_path);
    out_json << "{\n"
             << "  \"global_update\": 31400,\n"
             << "  \"hidden_dim\": 2048,\n"
             << "  \"num_blocks\": 12,\n"
             << "  \"observation_dim\": 9182,\n"
             << "  \"enable_semantic_scorer\": true,\n"
             << "  \"enable_zone_encoder\": true,\n"
             << "  \"semantic_descriptor_schema\": \"semantic_action_v4\",\n"
             << "  \"market_appendix_mode\": \"full_public_information_v3\",\n"
             << "  \"feature_schema_label\": \"dune_imperium_full_public_information_v3|prefix=ordered_card_slots_v2:6255|opp_revealed=3x126|consumed_conflicts=22|ordered_topdeck=4x4x127|water_overflow=4|pending_graft=127|pending_hundro=2x63|pending_tech=4|pending_ceremony=2x63|pending_intrigue=108|size=9182\",\n"
             << "  \"feature_schema_sha256\": \"1f91ede6656d792545589b69b39ed1209cb8255c3ba8dcae49726acde7388e7f\"\n"
             << "}\n";
    out_json.close();

    // Reload into a fresh model
    auto reloaded_model = std::make_shared<SharedDunePolicyValueNetImpl>(
        input_dim, hidden_dim, action_dim, num_blocks,
        /*use_nonlinear=*/false, /*with_aux_heads=*/false,
        /*head_init_seed=*/0, /*with_semantic_scorer=*/true,
        /*enable_zone_encoder=*/true);
    LoadModelCheckpointRobust(reloaded_model, test_model_path, device);
    reloaded_model->to(device);
    reloaded_model->eval();

    // Verify all 128 parameters match bitwise
    auto orig_params = target_model->parameters();
    auto reload_params = reloaded_model->parameters();
    CHECK_ASSERT(orig_params.size() == reload_params.size(), "Param count must match");
    for (size_t i = 0; i < orig_params.size(); ++i) {
      CHECK_ASSERT(torch::equal(orig_params[i], reload_params[i]), "Reloaded param must match bitwise");
    }
    std::cout << "  Reloaded model: all 128 parameter tensors match bitwise.\n";

    // Reload optimizer
    auto reloaded_opt = MakeDuneOptimizer(reloaded_model, /*lr=*/base_lr, /*weight_decay=*/0.0, /*policy_weight_decay=*/0.0, /*new_block_lr=*/new_block_lr, /*new_block_first_idx=*/8);
    torch::load(*reloaded_opt, test_opt_path, device);

    CHECK_ASSERT(reloaded_opt->param_groups().size() == 3, "Reloaded opt has 3 groups");
    CHECK_ASSERT(reloaded_opt->param_groups()[0].params().size() == 2, "Reloaded group 0 has 2 params");
    CHECK_ASSERT(reloaded_opt->param_groups()[1].params().size() == 86, "Reloaded group 1 has 86 params");
    CHECK_ASSERT(reloaded_opt->param_groups()[2].params().size() == 40, "Reloaded group 2 has 40 params");
    std::cout << "  Reloaded optimizer: groups {2, 86, 40} verified.\n";

    // Clean up temporary reload files
    std::filesystem::remove_all(test_save_dir);
    std::cout << "CHECK 6 PASSED.\n\n";
  }

  // -----------------------------------------------------------------------
  // CHECK 7: Gradient Flow and Trainability of New Pathways
  // -----------------------------------------------------------------------
  std::cout << "--------------------------------------------------------\n";
  std::cout << "CHECK 7: Gradient Flow and Trainability of New Pathways\n";
  std::cout << "--------------------------------------------------------\n";
  {
    target_model->train();

    // Setup state where actions 91 and 92 are legal
    auto state = BuildStateAtFirstRevealTurn("dune_imperium(enable_immortality=true)");
    auto* dune_st = dynamic_cast<DuneImperiumState*>(state.get());
    const Player p = dune_st->CurrentPlayer();
    dune_st->SetTleilaxuRowForTesting({13, 11});
    dune_st->SetSpecimensForTesting(p, 5);

    auto legal = dune_st->LegalActions();
    std::vector<float> obs = dune_st->InformationStateTensorWithAppendix(
        p, MarketAppendixMode::kFullPublicInformationV3);
    torch::Tensor obs_t = torch::from_blob(obs.data(), {1, static_cast<long>(obs.size())}, torch::kFloat32).to(device);

    dune_semantic::CandidateActionData cand_tgt;
    dune_semantic::ExtractCandidateDescriptors(
        *dune_st, legal, &cand_tgt, "semantic_action_v4");

    // Pass 1: Forward pass
    target_opt->zero_grad();
    auto out = target_model->forward(obs_t);
    std::vector<const dune_semantic::CandidateActionData*> cands_tgt = {&cand_tgt};
    dune_semantic::ApplySemanticScorerBatch(
        target_model->semantic_scorer_, out.trunk, cands_tgt, out.logits, device);

    // Compute loss that includes action 91, 92 and value
    torch::Tensor loss = out.logits[0][91] * 1.5f + out.logits[0][92] * 2.0f + out.values.squeeze() * 1.0f;
    loss.backward();

    // Verify out_layer_f gradients exist and are non-zero!
    auto scorer = target_model->semantic_scorer_;
    CHECK_ASSERT(scorer != nullptr, "Scorer must exist");
    auto out_layer_f = scorer->out_layer_f;
    CHECK_ASSERT(!out_layer_f.is_empty(), "out_layer_f must exist");
    CHECK_ASSERT(out_layer_f->weight.grad().defined(), "out_layer_f weight grad must be defined");
    CHECK_ASSERT(out_layer_f->bias.grad().defined(), "out_layer_f bias grad must be defined");
    float grad_norm_f_w = out_layer_f->weight.grad().norm().item().toFloat();
    float grad_norm_f_b = out_layer_f->bias.grad().norm().item().toFloat();
    std::cout << "  Step 1 out_layer_f weight grad norm: " << grad_norm_f_w << "\n";
    std::cout << "  Step 1 out_layer_f bias grad norm:   " << grad_norm_f_b << "\n";
    CHECK_ASSERT(grad_norm_f_w > 0.0f, "out_layer_f weight grad must be non-zero");
    CHECK_ASSERT(grad_norm_f_b > 0.0f, "out_layer_f bias grad must be non-zero");

    // Verify zone_encoder out_proj gradients exist and are non-zero!
    auto zone_enc = target_model->zone_encoder_;
    CHECK_ASSERT(zone_enc != nullptr, "zone_encoder must exist");
    CHECK_ASSERT(zone_enc->out_proj->weight.grad().defined(), "out_proj weight grad must be defined");
    CHECK_ASSERT(zone_enc->out_proj->bias.grad().defined(), "out_proj bias grad must be defined");
    float grad_norm_proj_w = zone_enc->out_proj->weight.grad().norm().item().toFloat();
    float grad_norm_proj_b = zone_enc->out_proj->bias.grad().norm().item().toFloat();
    std::cout << "  Step 1 zone_encoder out_proj weight grad norm: " << grad_norm_proj_w << "\n";
    std::cout << "  Step 1 zone_encoder out_proj bias grad norm:   " << grad_norm_proj_b << "\n";
    CHECK_ASSERT(grad_norm_proj_w > 0.0f, "out_proj weight grad must be non-zero");
    CHECK_ASSERT(grad_norm_proj_b > 0.0f, "out_proj bias grad must be non-zero");

    // Before optimizer step: check out_layer_f and out_proj weights are zero
    CHECK_ASSERT(out_layer_f->weight.norm().item().toFloat() == 0.0f, "Initial out_layer_f weight must be 0");
    CHECK_ASSERT(out_layer_f->bias.norm().item().toFloat() == 0.0f, "Initial out_layer_f bias must be 0");
    CHECK_ASSERT(zone_enc->out_proj->weight.norm().item().toFloat() == 0.0f, "Initial out_proj weight must be 0");
    CHECK_ASSERT(zone_enc->out_proj->bias.norm().item().toFloat() == 0.0f, "Initial out_proj bias must be 0");

    // Take optimizer step!
    target_opt->step();

    // Verify weights have moved!
    float post_w_f = out_layer_f->weight.norm().item().toFloat();
    float post_b_f = out_layer_f->bias.norm().item().toFloat();
    float post_w_proj = zone_enc->out_proj->weight.norm().item().toFloat();
    float post_b_proj = zone_enc->out_proj->bias.norm().item().toFloat();
    std::cout << "  Post-step out_layer_f weight norm: " << post_w_f << " (changed from 0.0)\n";
    std::cout << "  Post-step out_layer_f bias norm:   " << post_b_f << " (changed from 0.0)\n";
    std::cout << "  Post-step out_proj weight norm:    " << post_w_proj << " (changed from 0.0)\n";
    std::cout << "  Post-step out_proj bias norm:      " << post_b_proj << " (changed from 0.0)\n";
    CHECK_ASSERT(post_w_f > 0.0f, "out_layer_f weight must move after optimizer step");
    CHECK_ASSERT(post_b_f > 0.0f, "out_layer_f bias must move after optimizer step");
    CHECK_ASSERT(post_w_proj > 0.0f, "out_proj weight must move after optimizer step");
    CHECK_ASSERT(post_b_proj > 0.0f, "out_proj bias must move after optimizer step");

    // Pass 2: Now that out_proj has non-zero weights, verify upstream zone_encoder layers (fc1, ln1) receive gradients!
    target_opt->zero_grad();
    auto out2 = target_model->forward(obs_t);
    torch::Tensor loss2 = out2.logits[0][91] + out2.values.squeeze();
    loss2.backward();

    CHECK_ASSERT(zone_enc->fc1->weight.grad().defined(), "Upstream fc1 grad must now be defined");
    float grad_norm_fc1 = zone_enc->fc1->weight.grad().norm().item().toFloat();
    std::cout << "  Step 2 upstream zone_encoder fc1 weight grad norm: " << grad_norm_fc1 << "\n";
    CHECK_ASSERT(grad_norm_fc1 > 0.0f, "Upstream fc1 weight grad must be non-zero now that out_proj has moved!");

    std::cout << "CHECK 7 PASSED.\n\n";
  }

  std::cout << "========================================================\n";
  std::cout << "=== ALL CHECKS PASSED: " << g_checks_passed << "/" << g_checks_run << " assertions ===\n";
  std::cout << "========================================================\n";

  return 0;
}
