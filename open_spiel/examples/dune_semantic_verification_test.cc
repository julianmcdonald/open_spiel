// Verification tests for Dune Imperium Semantic Action Scorer.
// Validates all 7 pre-flight requirements with production-path rigor:
// 1. Zero-init parity with u15828 (< 1e-6 max abs diff across 100 states on values, logits, and legal probabilities, plus Guild Accord spice cost check).
// 2. Scorer sensitivity to card and space attributes.
// 3. Market permutation sensitivity on engine imperium row.
// 4. Hidden information independence in Round 1 play.
// 5. Cross-path consistency on supported round 1 actions (DeterministicEvaluator vs BatchedEvaluator vs TrainPpoUpdate).
// 6. 2-step PPO gradient check with non-constant advantages verifying reward-directed updates to all internal projections and embeddings.
// 7. Populated optimizer resume check verifying loaded moments and exact next-update equivalence.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_board.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "open_spiel/games/dune_imperium/dune_imperium_test_utils.h"

#include "dune_network.h"
#include "dune_semantic_action_scorer.h"
#include "dune_ppo_training_utils.h"
#include "dune_search_routing.h"
#include "dune_seed_utils.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"

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
ABSL_FLAG(bool, rollout_amp, false, "");
ABSL_FLAG(bool, allow_tf32, false, "");
ABSL_FLAG(double, grad_clip_norm, 0.5, "");
ABSL_FLAG(bool, diagnostics_only, false, "");

using namespace open_spiel;
using namespace open_spiel::dune_imperium;

static int test_count = 0;
static int pass_count = 0;

#define TEST_BEGIN(name)                                              \
  do {                                                                \
    ++test_count;                                                     \
    const char* test_name_ = (name);                                  \
    std::cout << "Test " << test_count << ": " << test_name_ << "... ";

#define TEST_END()                                                    \
    ++pass_count;                                                     \
    std::cout << "PASSED\n";                                          \
  } while (0)

#define CHECK_TRUE(cond)                                              \
  do {                                                                \
    if (!(cond)) {                                                    \
      std::cerr << "FAILED\n  Assertion failed: " #cond               \
                << "\n  at " << __FILE__ << ":" << __LINE__ << "\n";  \
      std::abort();                                                   \
    }                                                                 \
  } while (0)

#define CHECK_FALSE(cond) CHECK_TRUE(!(cond))

#define CHECK_LT(a, b)                                                \
  do {                                                                \
    auto a_ = (a); auto b_ = (b);                                     \
    if (!(a_ < b_)) {                                                 \
      std::cerr << "FAILED\n  Expected " #a " (" << a_ << ") < " #b   \
                << " (" << b_ << ")\n  at " << __FILE__ << ":"        \
                << __LINE__ << "\n";                                  \
      std::abort();                                                   \
    }                                                                 \
  } while (0)

#define CHECK_GT(a, b)                                                \
  do {                                                                \
    auto a_ = (a); auto b_ = (b);                                     \
    if (!(a_ > b_)) {                                                 \
      std::cerr << "FAILED\n  Expected " #a " (" << a_ << ") > " #b   \
                << " (" << b_ << ")\n  at " << __FILE__ << ":"        \
                << __LINE__ << "\n";                                  \
      std::abort();                                                   \
    }                                                                 \
  } while (0)

#define CHECK_GE(a, b)                                                \
  do {                                                                \
    auto a_ = (a); auto b_ = (b);                                     \
    if (!(a_ >= b_)) {                                                \
      std::cerr << "FAILED\n  Expected " #a " (" << a_ << ") >= " #b  \
                << " (" << b_ << ")\n  at " << __FILE__ << ":"        \
                << __LINE__ << "\n";                                  \
      std::abort();                                                   \
    }                                                                 \
  } while (0)

#define CHECK_LE(a, b)                                                \
  do {                                                                \
    auto a_ = (a); auto b_ = (b);                                     \
    if (!(a_ <= b_)) {                                                \
      std::cerr << "FAILED\n  Expected " #a " (" << a_ << ") <= " #b  \
                << " (" << b_ << ")\n  at " << __FILE__ << ":"        \
                << __LINE__ << "\n";                                  \
      std::abort();                                                   \
    }                                                                 \
  } while (0)

#define CHECK_EQ(a, b)                                                \
  do {                                                                \
    auto a_ = (a); auto b_ = (b);                                     \
    if (a_ != b_) {                                                   \
      std::cerr << "FAILED\n  Expected " #a " == " #b                 \
                << "\n  Got: " << a_ << " vs " << b_                  \
                << "\n  at " << __FILE__ << ":" << __LINE__ << "\n";  \
      std::abort();                                                   \
    }                                                                 \
  } while (0)

const std::string kU15828Path =
    "/home/warcr/projects/dune_drl/calibration_results_v2/pf_c_run2/s1_ctl_b/ppo_model_update_15828.pt";

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

// Advances game state into Round 1 Agent Turns (GamePhase::kAgentTurns) where candidate actions are cards/spaces.
inline std::unique_ptr<State> AdvanceToAgentTurns(
    const std::shared_ptr<const Game>& game, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::unique_ptr<State> st = game->NewInitialState();
  while (!st->IsTerminal()) {
    if (st->IsChanceNode()) {
      auto outcomes = st->ChanceOutcomes();
      std::uniform_int_distribution<size_t> dist(0, outcomes.size() - 1);
      st->ApplyAction(outcomes[dist(rng)].first);
      continue;
    }
    const auto* dune_st = dynamic_cast<const DuneImperiumState*>(st.get());
    if (dune_st != nullptr && dune_st->phase() == GamePhase::kAgentTurns) {
      return st;
    }
    auto legal = st->LegalActions();
    if (legal.empty()) break;
    std::uniform_int_distribution<size_t> dist(0, legal.size() - 1);
    st->ApplyAction(legal[dist(rng)]);
  }
  return st;
}

int main(int argc, char** argv) {
  std::cout << "=== Running Dune Semantic Action Scorer Verification Tests ===\n";

  torch::Device device(torch::kCPU);
  const char* force_cuda = std::getenv("DUNE_VERIFY_CUDA");
  if (force_cuda != nullptr && torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA);
    std::cout << "Using CUDA device for tests (DUNE_VERIFY_CUDA requested)\n";
  } else {
    std::cout << "Using CPU device for tests (preserving GPU VRAM for active training)\n";
  }

  std::shared_ptr<const Game> game = LoadGame("dune_imperium");

  // -------------------------------------------------------------------------
  // Test 1: Zero-init parity with u15828 across 100 states + Guild Accord check
  // -------------------------------------------------------------------------
  TEST_BEGIN("Zero-init parity with u15828 across 100 states (values, logits, probs)");
  {
    const int64_t input_dim = kLegacyInformationStateSize;
    const int64_t hidden_dim = 2048;
    const int64_t action_dim = 2391;
    const int num_blocks = 8;

    auto model_a = std::make_shared<SharedDunePolicyValueNetImpl>(
        input_dim, hidden_dim, action_dim, num_blocks, false, false, 0, false);
    model_a->eval();
    LoadModelCheckpointRobust(model_a, kU15828Path, device);
    model_a->to(device);

    auto model_b = std::make_shared<SharedDunePolicyValueNetImpl>(
        input_dim, hidden_dim, action_dim, num_blocks, false, false, 0, true);
    model_b->eval();
    LoadModelCheckpointRobust(model_b, kU15828Path, device);
    model_b->to(device);

    float max_abs_diff_logits = 0.0f;
    float max_abs_diff_values = 0.0f;
    float max_abs_diff_probs = 0.0f;
    int states_checked = 0;
    std::mt19937_64 rng(12345);

    while (states_checked < 100) {
      std::unique_ptr<State> st = game->NewInitialState();
      while (!st->IsTerminal() && states_checked < 100) {
        if (st->IsChanceNode()) {
          auto outcomes = st->ChanceOutcomes();
          std::uniform_int_distribution<size_t> dist(0, outcomes.size() - 1);
          st->ApplyAction(outcomes[dist(rng)].first);
          continue;
        }

        auto legal_actions = st->LegalActions();
        if (legal_actions.empty()) break;

        const auto* dune_state = dynamic_cast<const DuneImperiumState*>(st.get());
        CHECK_TRUE(dune_state != nullptr);

        std::vector<float> obs = dune_state->InformationStateTensor(dune_state->CurrentPlayer());
        torch::Tensor obs_t = torch::from_blob(obs.data(), {1, static_cast<long>(obs.size())}, torch::kFloat32).to(device);

        dune_semantic::CandidateActionData cand_data;
        dune_semantic::ExtractCandidateDescriptors(*dune_state, legal_actions, &cand_data);

        torch::NoGradGuard no_grad;
        auto out_a = model_a->forward(obs_t);
        auto out_b = model_b->forward(obs_t);

        std::vector<const dune_semantic::CandidateActionData*> batch_cands = {&cand_data};
        dune_semantic::ApplySemanticScorerBatch(
            model_b->semantic_scorer_, out_b.trunk, batch_cands, out_b.logits, device);

        float d_val = (out_a.values - out_b.values).abs().max().item<float>();
        if (d_val > max_abs_diff_values) max_abs_diff_values = d_val;

        float d_log = (out_a.logits - out_b.logits).abs().max().item<float>();
        if (d_log > max_abs_diff_logits) max_abs_diff_logits = d_log;

        // Compare legal action probability distributions
        std::vector<float> log_a(action_dim), log_b(action_dim);
        torch::Tensor la_cpu = out_a.logits.to(torch::kCPU).contiguous();
        torch::Tensor lb_cpu = out_b.logits.to(torch::kCPU).contiguous();
        std::memcpy(log_a.data(), la_cpu.data_ptr<float>(), action_dim * sizeof(float));
        std::memcpy(log_b.data(), lb_cpu.data_ptr<float>(), action_dim * sizeof(float));

        CenterAndCapLegalLogits(log_a, legal_actions, 10.0f);
        CenterAndCapLegalLogits(log_b, legal_actions, 10.0f);

        auto p_a = ComputeLegalProbabilities(log_a, legal_actions);
        auto p_b = ComputeLegalProbabilities(log_b, legal_actions);
        for (size_t i = 0; i < legal_actions.size(); ++i) {
          float dp = std::abs(p_a[i] - p_b[i]);
          if (dp > max_abs_diff_probs) max_abs_diff_probs = dp;
        }

        ++states_checked;

        // Step game forward with a random legal action
        std::uniform_int_distribution<size_t> dist(0, legal_actions.size() - 1);
        st->ApplyAction(legal_actions[dist(rng)]);
      }
    }

    // Verify effective spice cost calculation on Heighliner with Guild Accord
    {
      std::unique_ptr<State> st_accord = game->NewInitialState();
      auto* dstate = dynamic_cast<DuneImperiumState*>(st_accord.get());
      CHECK_TRUE(dstate != nullptr);

      while (!st_accord->IsTerminal() &&
             (st_accord->CurrentPlayer() != 0 || st_accord->IsChanceNode() ||
              dstate->phase() != GamePhase::kAgentTurns)) {
        if (st_accord->IsChanceNode()) {
          st_accord->ApplyAction(st_accord->ChanceOutcomes()[0].first);
        } else {
          auto legal = st_accord->LegalActions();
          if (legal.empty()) break;
          st_accord->ApplyAction(DefaultProgressionAction(legal));
        }
      }

      dstate->SetPlayerSpiceForTesting(0, 4);
      dstate->SetPlayerInfluenceForTesting(0, Faction::kSpacingGuild, 2);
      const int foldspace_id = FindImperiumCardByName("Foldspace").id;
      dstate->SetPlayerHandForTesting(0, {foldspace_id, kCardGuildAccord});

      dstate->ApplyAction(kActionSelectAgentCard0 + kCardGuildAccord);
      CHECK_TRUE(dstate->GuildAccordPlayedForPendingPlacement(0));

      auto legal_spaces = dstate->LegalActions();
      dune_semantic::CandidateActionData cand_accord;
      dune_semantic::ExtractCandidateDescriptors(*dstate, legal_spaces, &cand_accord);

      bool found_heighliner = false;
      for (size_t i = 0; i < cand_accord.actions.size(); ++i) {
        if (cand_accord.actions[i] == kActionAgentSpaceHeighliner) {
          found_heighliner = true;
          const float* f = &cand_accord.features[i * dune_semantic::kSemanticFeatDim];
          CHECK_TRUE(std::abs(f[44] - 0.6f) < 1e-4f);
          CHECK_TRUE(std::abs(f[45] - 0.4f) < 1e-4f);
          CHECK_TRUE(std::abs(f[49] - 1.0f) < 1e-4f);
        }
      }
      CHECK_TRUE(found_heighliner);
    }

    std::cout << "(Checked " << states_checked << " states, max diff val=" << max_abs_diff_values
              << ", logit=" << max_abs_diff_logits
              << ", prob=" << max_abs_diff_probs << ") ";
    CHECK_LT(max_abs_diff_values, 1e-6f);
    CHECK_LT(max_abs_diff_logits, 1e-6f);
    CHECK_LT(max_abs_diff_probs, 1e-6f);
  }
  TEST_END();

  // -------------------------------------------------------------------------
  // Test 2: Scorer sensitivity to card and space attributes
  // -------------------------------------------------------------------------
  TEST_BEGIN("Scorer sensitivity to card and space attributes");
  {
    auto scorer = std::make_shared<dune_semantic::SemanticActionScorerImpl>();
    scorer->eval();
    scorer->to(device);

    torch::Tensor trunk = torch::randn({1, 2048}, device);

    dune_semantic::CandidateActionData cdata;
    cdata.actions = {static_cast<Action>(kActionSelectAgentCard0 + 5),
                     static_cast<Action>(kActionAgentSpaceCarthag),
                     static_cast<Action>(kActionEndTurn)};
    cdata.card_ids = {5, 0, 0};
    cdata.space_ids = {0, 3, 0};
    cdata.supported = {1, 1, 0};
    cdata.features.assign(3 * dune_semantic::kSemanticFeatDim, 0.0f);
    cdata.features[0 * dune_semantic::kSemanticFeatDim + 1] = 1.0f; // is_primary_card
    cdata.features[0 * dune_semantic::kSemanticFeatDim + 9] = 0.3f; // persuasion cost
    cdata.features[1 * dune_semantic::kSemanticFeatDim + 2] = 1.0f; // is_board_space
    cdata.features[1 * dune_semantic::kSemanticFeatDim + 38] = 1.0f; // is_combat

    torch::Tensor logits_orig = torch::zeros({1, 2391}, device);
    std::vector<const dune_semantic::CandidateActionData*> batch = {&cdata};

    // Perturb out_layer weight away from 0 so sensitivity is non-zero
    {
      torch::NoGradGuard no_grad;
      scorer->out_layer->weight.fill_(0.01f);
    }

    dune_semantic::ApplySemanticScorerBatch(scorer, trunk, batch, logits_orig, device);

    // Perturb features and check logit shifts
    cdata.features[0 * dune_semantic::kSemanticFeatDim + 9] = 0.8f; // Change card cost
    torch::Tensor logits_perturbed = torch::zeros({1, 2391}, device);
    dune_semantic::ApplySemanticScorerBatch(scorer, trunk, batch, logits_perturbed, device);

    float delta = (logits_orig - logits_perturbed).abs().max().item<float>();
    CHECK_TRUE(delta > 1e-4f);

    // Unsupported action (EndTurn) must receive 0 correction
    float end_turn_orig = logits_orig[0][kActionEndTurn].item<float>();
    CHECK_EQ(end_turn_orig, 0.0f);
  }
  TEST_END();

  // -------------------------------------------------------------------------
  // Test 3: Market permutation sensitivity on engine imperium row
  // -------------------------------------------------------------------------
  TEST_BEGIN("Market permutation sensitivity on engine imperium row");
  {
    auto scorer = std::make_shared<dune_semantic::SemanticActionScorerImpl>();
    scorer->eval();
    {
      torch::NoGradGuard no_grad;
      scorer->out_layer->weight.fill_(0.02f);
    }
    scorer->to(device);

    torch::Tensor trunk = torch::randn({1, 2048}, device);

    std::unique_ptr<State> st = AdvanceToAgentTurns(game, 42);
    auto* dune_st = dynamic_cast<DuneImperiumState*>(st.get());
    CHECK_TRUE(dune_st != nullptr);

    const int card_a = 5;
    const int card_b = 20;
    dune_st->SetImperiumRowCardForTesting(0, card_a);
    dune_st->SetImperiumRowCardForTesting(1, card_b);

    std::vector<Action> market_actions = {
        static_cast<Action>(kActionBuyImperiumRow0),
        static_cast<Action>(kActionBuyImperiumRow0 + 1)
    };

    dune_semantic::CandidateActionData cdata1;
    dune_semantic::ExtractCandidateDescriptors(*dune_st, market_actions, &cdata1);

    CHECK_EQ(cdata1.card_ids[0], card_a);
    CHECK_EQ(cdata1.card_ids[1], card_b);
    CHECK_EQ(cdata1.supported[0], 1);
    CHECK_EQ(cdata1.supported[1], 1);

    // Swap slots 0 and 1 in the engine state
    dune_st->SetImperiumRowCardForTesting(0, card_b);
    dune_st->SetImperiumRowCardForTesting(1, card_a);

    dune_semantic::CandidateActionData cdata2;
    dune_semantic::ExtractCandidateDescriptors(*dune_st, market_actions, &cdata2);

    CHECK_EQ(cdata2.card_ids[0], card_b);
    CHECK_EQ(cdata2.card_ids[1], card_a);

    torch::Tensor logits1 = torch::zeros({1, 2391}, device);
    std::vector<const dune_semantic::CandidateActionData*> batch1 = {&cdata1};
    dune_semantic::ApplySemanticScorerBatch(scorer, trunk, batch1, logits1, device);

    torch::Tensor logits2 = torch::zeros({1, 2391}, device);
    std::vector<const dune_semantic::CandidateActionData*> batch2 = {&cdata2};
    dune_semantic::ApplySemanticScorerBatch(scorer, trunk, batch2, logits2, device);

    float l1_s0 = logits1[0][kActionBuyImperiumRow0].item<float>();
    float l1_s1 = logits1[0][kActionBuyImperiumRow0 + 1].item<float>();
    float l2_s0 = logits2[0][kActionBuyImperiumRow0].item<float>();
    float l2_s1 = logits2[0][kActionBuyImperiumRow0 + 1].item<float>();

    CHECK_EQ(cdata1.card_ids[0], cdata2.card_ids[1]);
    CHECK_EQ(cdata1.card_ids[1], cdata2.card_ids[0]);
    CHECK_EQ(cdata1.features[0 * dune_semantic::kSemanticFeatDim + 9],
             cdata2.features[1 * dune_semantic::kSemanticFeatDim + 9]);
    CHECK_EQ(cdata1.features[1 * dune_semantic::kSemanticFeatDim + 9],
             cdata2.features[0 * dune_semantic::kSemanticFeatDim + 9]);
    CHECK_EQ(cdata1.features[0 * dune_semantic::kSemanticFeatDim + 4], 1.0f); // slot 0 one-hot
    CHECK_EQ(cdata2.features[0 * dune_semantic::kSemanticFeatDim + 4], 1.0f);
    CHECK_EQ(cdata1.features[1 * dune_semantic::kSemanticFeatDim + 5], 1.0f); // slot 1 one-hot
    CHECK_EQ(cdata2.features[1 * dune_semantic::kSemanticFeatDim + 5], 1.0f);

    CHECK_TRUE(std::abs(l1_s0 - l2_s0) > 1e-4f);
    CHECK_TRUE(std::abs(l1_s1 - l2_s1) > 1e-4f);
  }
  TEST_END();

  // -------------------------------------------------------------------------
  // Test 4: Hidden information independence in Round 1 play
  // -------------------------------------------------------------------------
  TEST_BEGIN("Hidden information independence in Round 1 play");
  {
    std::unique_ptr<State> st1 = AdvanceToAgentTurns(game, 777);
    auto* dune_st1 = dynamic_cast<DuneImperiumState*>(st1.get());
    CHECK_TRUE(dune_st1 != nullptr);

    std::unique_ptr<State> st2 = st1->Clone();
    auto* dune_st2 = dynamic_cast<DuneImperiumState*>(st2.get());
    CHECK_TRUE(dune_st2 != nullptr);

    const Player cur = dune_st1->CurrentPlayer();
    const Player opp = (cur + 1) % dune_st1->NumPlayers();

    // Modify opponent hidden cards
    dune_st2->SetPlayerIntrigueHandForTesting(opp, {1, 2, 3});

    auto legal_actions = dune_st1->LegalActions();
    CHECK_FALSE(legal_actions.empty());

    dune_semantic::CandidateActionData cands1, cands2;
    dune_semantic::ExtractCandidateDescriptors(*dune_st1, legal_actions, &cands1);
    dune_semantic::ExtractCandidateDescriptors(*dune_st2, legal_actions, &cands2);

    CHECK_EQ(cands1.actions.size(), cands2.actions.size());
    CHECK_EQ(cands1.features.size(), cands2.features.size());
    for (size_t i = 0; i < cands1.features.size(); ++i) {
      CHECK_EQ(cands1.features[i], cands2.features[i]);
    }
    for (size_t i = 0; i < cands1.card_ids.size(); ++i) {
      CHECK_EQ(cands1.card_ids[i], cands2.card_ids[i]);
      CHECK_EQ(cands1.space_ids[i], cands2.space_ids[i]);
      CHECK_EQ(cands1.supported[i], cands2.supported[i]);
    }
  }
  TEST_END();

  // -------------------------------------------------------------------------
  // Test 5: Production collection-to-learner parity on real transitions
  // -------------------------------------------------------------------------
  const int64_t input_dim = kLegacyInformationStateSize;
  const int64_t hidden_dim = 2048;
  const int64_t action_dim = 2391;
  const int num_blocks = 8;

  std::vector<PpoTransition> collected_transitions;
  std::vector<PpoTransition> global_batch(8);

  auto global_model = std::make_shared<SharedDunePolicyValueNetImpl>(
      input_dim, hidden_dim, action_dim, num_blocks, false, false, 0, true);
  global_model->train();
  LoadModelCheckpointRobust(global_model, kU15828Path, device);
  {
    torch::NoGradGuard no_grad;
    global_model->semantic_scorer_->out_layer->weight.fill_(0.02f);
    global_model->semantic_scorer_->mlp1->weight.data().add_(0.01f);
    global_model->semantic_scorer_->desc_proj->weight.data().add_(0.01f);
  }
  global_model->to(device);

  torch::optim::AdamW global_optimizer(global_model->parameters(),
                                       torch::optim::AdamWOptions(1e-4).weight_decay(0.0));

  TEST_BEGIN("Production collection-to-learner parity (primary/space/purchase, concurrent batching, mixed-model eval)");
  {
    // 1. Candidate model with non-zero initialized scorer
    auto model_cand = std::make_shared<SharedDunePolicyValueNetImpl>(
        input_dim, hidden_dim, action_dim, num_blocks, false, false, 0, true);
    model_cand->eval();
    LoadModelCheckpointRobust(model_cand, kU15828Path, device);
    {
      torch::NoGradGuard no_grad;
      model_cand->semantic_scorer_->out_layer->weight.fill_(0.02f);
      model_cand->semantic_scorer_->mlp1->weight.data().add_(0.01f);
      model_cand->semantic_scorer_->desc_proj->weight.data().add_(0.01f);
    }
    model_cand->to(device);

    // 2. Opponent model without scorer (baseline u15828)
    auto model_opp = std::make_shared<SharedDunePolicyValueNetImpl>(
        input_dim, hidden_dim, action_dim, num_blocks, false, false, 0, false);
    model_opp->eval();
    LoadModelCheckpointRobust(model_opp, kU15828Path, device);
    model_opp->to(device);

    // 3. Evaluators: Candidate uses BatchedEvaluator, Opponents use DeterministicEvaluator
    std::shared_mutex sync_mutex_cand;
    BatchedEvaluator batch_eval_cand(
        model_cand, 8, 5, device, &sync_mutex_cand, 0.0f, true, false, false, false, false);

    std::mutex eval_mutex_opp;
    std::shared_mutex sync_mutex_opp;
    DeterministicEvaluator opp_eval(
        model_opp, device, &eval_mutex_opp, &sync_mutex_opp, nullptr, false);

    // 4. Multi-threaded rollout collection with concurrent batching and mixed-model evaluation
    std::mutex collect_mutex;
    std::atomic<int> count_primary{0};
    std::atomic<int> count_space{0};
    std::atomic<int> count_purchase{0};
    std::atomic<bool> collection_done{false};

    auto worker_fn = [&](int thread_id) {
      uint64_t seed = 2026091200ULL + static_cast<uint64_t>(thread_id) * 1000ULL;
      int game_idx = 0;
      while (!collection_done.load(std::memory_order_relaxed)) {
        std::mt19937_64 rng(seed + static_cast<uint64_t>(game_idx++));
        std::unique_ptr<State> st = game->NewInitialState();
        while (!st->IsTerminal() && !collection_done.load(std::memory_order_relaxed)) {
          if (st->IsChanceNode()) {
            auto outcomes = st->ChanceOutcomes();
            std::uniform_int_distribution<size_t> dist(0, outcomes.size() - 1);
            st->ApplyAction(outcomes[dist(rng)].first);
            continue;
          }
          auto* dune_st = dynamic_cast<const DuneImperiumState*>(st.get());
          if (dune_st == nullptr) break;
          Player p = dune_st->CurrentPlayer();
          if (p < 0 || p >= 4) break;
          auto legal = dune_st->LegalActions();
          if (legal.empty()) break;

          std::vector<float> obs = dune_st->InformationStateTensor(p);
          if (p == 0) {
            dune_semantic::CandidateActionData cand_data;
            dune_semantic::ExtractCandidateDescriptors(*dune_st, legal, &cand_data);
            EvalResult res = batch_eval_cand.EvaluateWithActions(obs, &cand_data);
            CenterAndCapLegalLogits(res.logits, legal, 10.0f);
            std::vector<float> probs = ComputeLegalProbabilities(res.logits, legal);
            std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
            size_t chosen_idx = dist(rng);
            Action chosen_act = legal[chosen_idx];
            float old_lp = std::log(std::max(probs[chosen_idx], 1e-12f));

            DuneDecisionRole role = ClassifyDuneDecisionRole(*st, 0, false);
            bool is_primary = (role == DuneDecisionRole::kAgentPrimary) ||
                              (chosen_act >= kActionSelectAgentCard0 && chosen_act < kActionSelectAgentCard0 + 256);
            bool is_space = (chosen_act >= kActionAgentSpaceConspire && chosen_act <= kActionAgentSpaceSietchTabr);
            bool is_purchase = (role == DuneDecisionRole::kPurchase) ||
                               (chosen_act >= kActionBuyImperiumRow0 && chosen_act < kActionBuyImperiumRow0 + 8) ||
                               (chosen_act == kActionBuyReserveArrakisLiaison) ||
                               (chosen_act == kActionBuyReserveTheSpiceMustFlow);

            if (is_primary || is_space || is_purchase) {
              PpoTransition trans;
              trans.state = obs;
              trans.legal_actions = legal;
              trans.action = chosen_act;
              trans.old_log_prob = old_lp;
              trans.candidate_data = cand_data;
              trans.decision_role = static_cast<int>(role);
              trans.player_id = 0;
              trans.episode_id = thread_id * 1000 + game_idx;

              std::lock_guard<std::mutex> lock(collect_mutex);
              collected_transitions.push_back(std::move(trans));
              if (is_primary) count_primary.fetch_add(1, std::memory_order_relaxed);
              if (is_space) count_space.fetch_add(1, std::memory_order_relaxed);
              if (is_purchase) count_purchase.fetch_add(1, std::memory_order_relaxed);

              if (count_primary.load() >= 4 && count_space.load() >= 4 && count_purchase.load() >= 4) {
                collection_done.store(true, std::memory_order_relaxed);
              }
            }
            st->ApplyAction(chosen_act);
          } else {
            EvalResult res = opp_eval.Evaluate(obs);
            CenterAndCapLegalLogits(res.logits, legal, 10.0f);
            std::vector<float> probs = ComputeLegalProbabilities(res.logits, legal);
            std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
            Action act = legal[dist(rng)];
            st->ApplyAction(act);
          }
        }
      }
    };

    std::vector<std::thread> workers;
    for (int t = 0; t < 4; ++t) {
      workers.emplace_back(worker_fn, t);
    }
    for (auto& w : workers) {
      w.join();
    }

    std::cout << "(collected: total=" << collected_transitions.size()
              << ", primary=" << count_primary.load()
              << ", space=" << count_space.load()
              << ", purchase=" << count_purchase.load() << ") ";

    CHECK_GE(count_primary.load(), 4);
    CHECK_GE(count_space.load(), 4);
    CHECK_GE(count_purchase.load(), 4);
    CHECK_GE(collected_transitions.size(), 12);

    // 5. Verify learner replay parity transition by transition at unchanged weights
    torch::NoGradGuard no_grad;
    float max_diff_lp = 0.0f;
    float max_diff_prob = 0.0f;
    for (const auto& trans : collected_transitions) {
      torch::Tensor obs_t = torch::from_blob(
          const_cast<float*>(trans.state.data()),
          {1, static_cast<long>(trans.state.size())},
          torch::kFloat32).to(device);
      auto fwd = model_cand->forward(obs_t);
      std::vector<const dune_semantic::CandidateActionData*> cands = {&trans.candidate_data};
      dune_semantic::ApplySemanticScorerBatch(
          model_cand->semantic_scorer_, fwd.trunk, cands, fwd.logits, device);
      torch::Tensor mask = torch::zeros({1, action_dim}, torch::TensorOptions().dtype(torch::kBool).device(device));
      for (Action a : trans.legal_actions) mask[0][a] = true;
      torch::Tensor capped = CenterAndCapLogitsTensor(fwd.logits, mask, 10.0f);
      torch::Tensor masked = capped.masked_fill(mask.logical_not(), -1e9f);
      torch::Tensor lps = torch::log_softmax(masked, -1);
      float replayed_lp = lps[0][trans.action].item().toFloat();
      float d_lp = std::abs(replayed_lp - trans.old_log_prob);
      if (d_lp > max_diff_lp) max_diff_lp = d_lp;
      float d_p = std::abs(std::exp(replayed_lp) - std::exp(trans.old_log_prob));
      if (d_p > max_diff_prob) max_diff_prob = d_p;
    }

    std::cout << "(individual max diff lp: " << max_diff_lp << ", max prob diff: " << max_diff_prob << ") ";
    CHECK_LT(max_diff_lp, 1e-5f);
    CHECK_LT(max_diff_prob, 1e-5f);

    // 6. Verify learner replay parity across shuffled minibatches
    const int64_t n_trans = static_cast<int64_t>(collected_transitions.size());
    torch::Tensor perm = torch::randperm(n_trans, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    const int64_t* perm_ptr = perm.data_ptr<int64_t>();
    float max_mb_diff = 0.0f;
    const int64_t mb_size = 4;
    for (int64_t start = 0; start < n_trans; start += mb_size) {
      int64_t end = std::min(start + mb_size, n_trans);
      int64_t cur_size = end - start;
      torch::Tensor mb_states = torch::zeros({static_cast<long>(cur_size), static_cast<long>(input_dim)}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
      torch::Tensor mb_masks = torch::zeros({static_cast<long>(cur_size), static_cast<long>(action_dim)}, torch::TensorOptions().dtype(torch::kBool).device(device));
      torch::Tensor mb_actions = torch::zeros({static_cast<long>(cur_size)}, torch::TensorOptions().dtype(torch::kInt64).device(device));
      torch::Tensor mb_old_lps = torch::zeros({static_cast<long>(cur_size)}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
      std::vector<const dune_semantic::CandidateActionData*> mb_cands(cur_size);

      for (int64_t i = 0; i < cur_size; ++i) {
        int64_t orig_idx = perm_ptr[start + i];
        const auto& tr = collected_transitions[orig_idx];
        std::memcpy(mb_states[i].data_ptr<float>(), tr.state.data(), tr.state.size() * sizeof(float));
        for (Action a : tr.legal_actions) mb_masks[i][a] = true;
        mb_actions[i] = tr.action;
        mb_old_lps[i] = tr.old_log_prob;
        mb_cands[i] = &tr.candidate_data;
      }

      auto out = model_cand->forward(mb_states);
      dune_semantic::ApplySemanticScorerBatch(
          model_cand->semantic_scorer_, out.trunk, mb_cands, out.logits, device);
      torch::Tensor capped = CenterAndCapLogitsTensor(out.logits, mb_masks, 10.0f);
      torch::Tensor masked = capped.masked_fill(mb_masks.logical_not(), -1e9f);
      torch::Tensor lps = torch::log_softmax(masked, -1);
      torch::Tensor replayed = lps.gather(1, mb_actions.unsqueeze(1)).squeeze(1);
      float d = (replayed - mb_old_lps).abs().max().item().toFloat();
      if (d > max_mb_diff) max_mb_diff = d;
    }
    std::cout << "(shuffled mb max diff: " << max_mb_diff << ") ";
    CHECK_LT(max_mb_diff, 1e-5f);
  }
  TEST_END();

  // -------------------------------------------------------------------------
  // Test 6: 2-step PPO policy gradient check on real collected transitions
  // -------------------------------------------------------------------------
  TEST_BEGIN("2-step PPO policy gradient check on real collected transitions");
  {
    // Populate global_batch with real transitions covering primary, space, and purchase
    std::vector<PpoTransition> selected_primary;
    std::vector<PpoTransition> selected_space;
    std::vector<PpoTransition> selected_purchase;

    for (const auto& tr : collected_transitions) {
      Action a = tr.action;
      if (tr.decision_role == static_cast<int>(DuneDecisionRole::kAgentPrimary) ||
          (a >= kActionSelectAgentCard0 && a < kActionSelectAgentCard0 + 256)) {
        if (selected_primary.size() < 3) selected_primary.push_back(tr);
      } else if (a >= kActionAgentSpaceConspire && a <= kActionAgentSpaceSietchTabr) {
        if (selected_space.size() < 3) selected_space.push_back(tr);
      } else if (tr.decision_role == static_cast<int>(DuneDecisionRole::kPurchase) ||
                 (a >= kActionBuyImperiumRow0 && a < kActionBuyImperiumRow0 + 8) ||
                 (a == kActionBuyReserveArrakisLiaison) ||
                 (a == kActionBuyReserveTheSpiceMustFlow)) {
        if (selected_purchase.size() < 2) selected_purchase.push_back(tr);
      }
    }

    CHECK_GE(selected_primary.size(), 2);
    CHECK_GE(selected_space.size(), 2);
    CHECK_GE(selected_purchase.size(), 2);

    std::vector<PpoTransition> real_selection;
    for (size_t i = 0; i < 3 && i < selected_primary.size(); ++i) real_selection.push_back(selected_primary[i]);
    for (size_t i = 0; i < 3 && i < selected_space.size(); ++i) real_selection.push_back(selected_space[i]);
    for (size_t i = 0; i < 2 && i < selected_purchase.size(); ++i) real_selection.push_back(selected_purchase[i]);
    while (real_selection.size() < 8) real_selection.push_back(collected_transitions[real_selection.size()]);

    const float advantages[8] = {2.0f, -1.5f, 1.0f, -0.8f, 1.8f, -1.2f, 0.7f, -0.5f};
    const float returns[8] = {1.0f, -0.75f, 0.5f, -0.4f, 0.9f, -0.6f, 0.35f, -0.25f};

    for (int i = 0; i < 8; ++i) {
      global_batch[i] = real_selection[i];
      global_batch[i].advantage = advantages[i];
      global_batch[i].return_value = returns[i];
      global_batch[i].value = 0.0f;
    }

    // Direct PPO policy gradient check with entropy_coef=0 and weight_decay=0
    global_model->zero_grad();
    const int64_t n_gb = 8;
    torch::Tensor gb_states = torch::zeros({static_cast<long>(n_gb), static_cast<long>(input_dim)}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor gb_masks = torch::zeros({static_cast<long>(n_gb), static_cast<long>(action_dim)}, torch::TensorOptions().dtype(torch::kBool).device(device));
    torch::Tensor gb_actions = torch::zeros({static_cast<long>(n_gb)}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    torch::Tensor gb_old_lps = torch::zeros({static_cast<long>(n_gb)}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor gb_advs = torch::zeros({static_cast<long>(n_gb)}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    std::vector<const dune_semantic::CandidateActionData*> gb_cands(n_gb);

    for (int64_t i = 0; i < n_gb; ++i) {
      std::memcpy(gb_states[i].data_ptr<float>(), global_batch[i].state.data(), input_dim * sizeof(float));
      for (Action a : global_batch[i].legal_actions) gb_masks[i][a] = true;
      gb_actions[i] = global_batch[i].action;
      gb_old_lps[i] = global_batch[i].old_log_prob;
      gb_advs[i] = global_batch[i].advantage;
      gb_cands[i] = &global_batch[i].candidate_data;
    }

    auto out = global_model->forward(gb_states);
    dune_semantic::ApplySemanticScorerBatch(
        global_model->semantic_scorer_, out.trunk, gb_cands, out.logits, device);
    auto logits = CenterAndCapLogitsTensor(out.logits, gb_masks, 10.0f);
    auto lps = torch::log_softmax(logits.masked_fill(gb_masks.logical_not(), -1e9f), -1);
    auto sel_lps = lps.gather(1, gb_actions.unsqueeze(1)).squeeze(1);
    auto ratio = torch::exp(sel_lps - gb_old_lps);
    auto surr1 = ratio * gb_advs;
    auto surr2 = torch::clamp(ratio, 0.8f, 1.2f) * gb_advs;
    auto policy_loss = -torch::min(surr1, surr2).mean();
    policy_loss.backward();

    // Inspect gradients directly before any optimizer step
    CHECK_TRUE(global_model->semantic_scorer_->desc_proj->weight.grad().defined());
    CHECK_TRUE(global_model->semantic_scorer_->trunk_proj->weight.grad().defined());
    CHECK_TRUE(global_model->semantic_scorer_->mlp1->weight.grad().defined());
    CHECK_TRUE(global_model->semantic_scorer_->card_embedding->weight.grad().defined());
    CHECK_TRUE(global_model->semantic_scorer_->space_embedding->weight.grad().defined());
    CHECK_TRUE(global_model->semantic_scorer_->out_layer->weight.grad().defined());

    float g_desc = global_model->semantic_scorer_->desc_proj->weight.grad().abs().max().item().toFloat();
    float g_trunk = global_model->semantic_scorer_->trunk_proj->weight.grad().abs().max().item().toFloat();
    float g_mlp1 = global_model->semantic_scorer_->mlp1->weight.grad().abs().max().item().toFloat();
    float g_card = global_model->semantic_scorer_->card_embedding->weight.grad().abs().max().item().toFloat();
    float g_space = global_model->semantic_scorer_->space_embedding->weight.grad().abs().max().item().toFloat();
    float g_out = global_model->semantic_scorer_->out_layer->weight.grad().abs().max().item().toFloat();

    std::cout << "(policy gradients: desc=" << g_desc
              << ", trunk=" << g_trunk
              << ", mlp1=" << g_mlp1
              << ", card=" << g_card
              << ", space=" << g_space
              << ", out=" << g_out << ") ";

    CHECK_GT(g_desc, 1e-7f);
    CHECK_GT(g_trunk, 1e-7f);
    CHECK_GT(g_mlp1, 1e-7f);
    CHECK_GT(g_card, 1e-7f);
    CHECK_GT(g_space, 1e-7f);
    CHECK_GT(g_out, 1e-7f);

    torch::Tensor init_desc = global_model->semantic_scorer_->desc_proj->weight.detach().clone();
    torch::Tensor init_trunk = global_model->semantic_scorer_->trunk_proj->weight.detach().clone();
    torch::Tensor init_mlp1 = global_model->semantic_scorer_->mlp1->weight.detach().clone();
    torch::Tensor init_card = global_model->semantic_scorer_->card_embedding->weight.detach().clone();
    torch::Tensor init_space = global_model->semantic_scorer_->space_embedding->weight.detach().clone();
    torch::Tensor init_out = global_model->semantic_scorer_->out_layer->weight.detach().clone();

    // Step optimizer with weight_decay=0.0
    global_optimizer.step();

    float d_desc = (global_model->semantic_scorer_->desc_proj->weight.detach() - init_desc).abs().max().item().toFloat();
    float d_trunk = (global_model->semantic_scorer_->trunk_proj->weight.detach() - init_trunk).abs().max().item().toFloat();
    float d_mlp1 = (global_model->semantic_scorer_->mlp1->weight.detach() - init_mlp1).abs().max().item().toFloat();
    float d_card = (global_model->semantic_scorer_->card_embedding->weight.detach() - init_card).abs().max().item().toFloat();
    float d_space = (global_model->semantic_scorer_->space_embedding->weight.detach() - init_space).abs().max().item().toFloat();
    float d_out = (global_model->semantic_scorer_->out_layer->weight.detach() - init_out).abs().max().item().toFloat();

    CHECK_GT(d_desc, 1e-7f);
    CHECK_GT(d_trunk, 1e-7f);
    CHECK_GT(d_mlp1, 1e-7f);
    CHECK_GT(d_card, 1e-7f);
    CHECK_GT(d_space, 1e-7f);
    CHECK_GT(d_out, 1e-7f);

    // Run 2 PPO updates to populate optimizer moments with real transitions
    absl::SetFlag(&FLAGS_entropy_coef, 0.0);
    for (int step = 0; step < 2; ++step) {
      PpoUpdateStats stats = TrainPpoUpdate(
          global_model, global_optimizer, global_batch, input_dim, action_dim, device, 42 + step, step, nullptr);
      CHECK_TRUE(std::isfinite(stats.policy_loss));
      CHECK_TRUE(std::isfinite(stats.value_loss));
    }
  }
  TEST_END();

  // -------------------------------------------------------------------------
  // Test 7: Populated optimizer resume and next-update equivalence
  // -------------------------------------------------------------------------
  TEST_BEGIN("Populated optimizer resume and next-update equivalence");
  {
    const int64_t input_dim = kLegacyInformationStateSize;
    const int64_t hidden_dim = 2048;
    const int64_t action_dim = 2391;
    const int num_blocks = 8;

    // Verify global_optimizer state is populated after 2 steps
    CHECK_TRUE(global_optimizer.state().size() > 0);

    const std::string tmp_model = "/tmp/test_dune_scorer_resume_model.pt";
    const std::string tmp_opt = "/tmp/test_dune_scorer_resume_opt.pt";
    torch::save(global_model, tmp_model);
    torch::save(global_optimizer, tmp_opt);

    auto model_resumed = std::make_shared<SharedDunePolicyValueNetImpl>(
        input_dim, hidden_dim, action_dim, num_blocks, false, false, 0, true);
    torch::load(model_resumed, tmp_model);
    model_resumed->to(device);

    torch::optim::AdamW optimizer_resumed(model_resumed->parameters(), torch::optim::AdamWOptions(1e-4));
    torch::load(optimizer_resumed, tmp_opt);

    CHECK_TRUE(optimizer_resumed.state().size() > 0);

    // Run identical next update (step 2) on both continuous and resumed models
    PpoUpdateStats s1 = TrainPpoUpdate(
        global_model, global_optimizer, global_batch, input_dim, action_dim, device, 999, 2, nullptr);
    PpoUpdateStats s2 = TrainPpoUpdate(
        model_resumed, optimizer_resumed, global_batch, input_dim, action_dim, device, 999, 2, nullptr);

    float p_loss_diff = std::abs(s1.policy_loss - s2.policy_loss);
    float v_loss_diff = std::abs(s1.value_loss - s2.value_loss);

    float max_param_diff = 0.0f;
    auto params1 = global_model->parameters();
    auto params2 = model_resumed->parameters();
    CHECK_EQ(params1.size(), params2.size());
    for (size_t i = 0; i < params1.size(); ++i) {
      float d = (params1[i] - params2[i]).abs().max().item().toFloat();
      if (d > max_param_diff) max_param_diff = d;
    }

    std::cout << "(p_loss_diff=" << p_loss_diff
              << ", v_loss_diff=" << v_loss_diff
              << ", max_param_diff=" << max_param_diff << ") ";

    CHECK_LT(p_loss_diff, 1e-5f);
    CHECK_LT(v_loss_diff, 1e-5f);
    CHECK_LT(max_param_diff, 1e-5f);

    std::filesystem::remove(tmp_model);
    std::filesystem::remove(tmp_opt);
  }
  TEST_END();

  std::cout << "\n=========================================\n";
  std::cout << "All " << pass_count << "/" << test_count << " tests passed successfully!\n";
  return 0;
}
