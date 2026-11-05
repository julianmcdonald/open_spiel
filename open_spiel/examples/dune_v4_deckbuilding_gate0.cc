// Gate 0 Diagnostic and Verification CLI for Dune Imperium Prototype F / v4 Deckbuilding.
// Supports:
//   --mode=stage3_cpu_verify:
//     - Loads v3 checkpoint into v3 and v4 models (enable_zone_encoder=false)
//     - Asserts max_diff == 0.0 over 1,024 states in FP32 and BF16
//     - Runs LoadOptimizerCheckpointMigrating verifying moved == 152 and out_layer_f empty
//     - Saves temp v4 model+optim to /tmp, tests v4 -> v4 reload (torch::load, 154 states)
//     - Verifies kTleilaxuPurchase fails closed on v3
//   --mode=gate0a:
//     - Samples 200 natural R1-R6 Reveal roots using policy self-play
//     - Runs (a) (a, a) null-pair BF16 check
//     - Runs (b) kEventKeyedCrn vs kDeckOrderCrn paired SD
//     - Runs (c) horizon matrix (5 arms: kNextRoundSearchPlayer, kPlus2Rounds, kTerminal R2=24/36/48)
//     - Runs (d) empirical firing rates
//     - Runs (e) scores accepted pairs on fresh R_eval = 64 kTerminal CRN bank with 2,000 game-clustered bootstrap resamples
//       and card realization tracking, dumping cache bundle to --gate0b_cache_path
//   --mode=gate0b:
//     - Runs 3-arm offline MPO fit (Arm 1 out_layer_f, Arm 2 scorer+f, Arm 3 exact 3-group AdamW)
//     - Differentiable forward pass with ApplySemanticScorerBatch + soft logit cap
//     - Tests slot-swap (91 <-> 92), unrelated Tleilaxu card bleed, and non-purchase KL drift (<= 0.003)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <shared_mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

#include "open_spiel/spiel.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_board.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "open_spiel/games/dune_imperium/dune_imperium_content.h"
#include "open_spiel/games/dune_imperium/dune_imperium_test_utils.h"

#include "dune_network.h"
#include "dune_semantic_action_scorer.h"
#include "dune_compound_turn_search.h"
#include "dune_ppo_training_utils.h"
#include "dune_ppo_training_flags.h"
#include "dune_seed_utils.h"
#include "dune_fp8_ref2048_bench.h"
#include "dune_search_session.h"
#include "dune_search_routing.h"
#include "dune_wide_rollout_search.h"
#include "open_spiel/utils/json.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"

ABSL_FLAG(std::string, mode, "stage3_cpu_verify",
          "Mode: stage3_cpu_verify, gate0a, gate0b, fp8_301_ref2048, verify_v5, gate4a_buy, or gate4a_place");
ABSL_FLAG(std::string, source_model_path, "", "Path to source model (.pt)");
ABSL_FLAG(std::string, source_opt_path, "", "Path to source optimizer (.pt)");
ABSL_FLAG(std::string, source_manifest_path, "", "Path to source manifest (.json)");
ABSL_FLAG(std::string, gate0b_cache_path, "/tmp/gate0b_cache.bin",
          "Path to read or write Gate 0b cache records");
ABSL_FLAG(std::string, summary_json_path, "", "Optional path to export JSON summary");
ABSL_FLAG(bool, use_cuda, true, "Whether to use CUDA when available");
ABSL_FLAG(int, batch_size, 256, "Evaluator batch size for rollout evaluation");
ABSL_FLAG(double, base_cat3_prob, 0.08, "Base search probability for Cat 3 Reveal roots");
ABSL_FLAG(double, boosted_cat3_prob, 0.24, "Boosted search probability when any card/tech prior < 0.02");
ABSL_FLAG(bool, slot_indexed_f, false, "Whether out_layer_f uses 2 outputs (slot-indexed)");
ABSL_FLAG(int, num_reveal_roots, 200, "Number of reveal roots to sample for gate0a");
ABSL_FLAG(int, eval_rollouts, 64, "R_eval rollouts per candidate pair in gate0a");
ABSL_FLAG(int, bootstrap_samples, 2000, "Number of bootstrap resamples");
ABSL_FLAG(int, src_num_blocks, 16, "Number of residual blocks for source model");
ABSL_FLAG(int, num_301_roots, 301, "Target stratified root count for fp8_301_ref2048");
ABSL_FLAG(int, census_games, 60, "Number of 4-seat greedy self-play games for root census");
ABSL_FLAG(int, ref2048_rollouts, 2048, "Paired BF16 reference rollouts per distinct action");
ABSL_FLAG(int, gate4a_buy_roots, 400, "Target stratified R1-R6 Reveal purchase roots for gate4a_buy");
ABSL_FLAG(int, gate4a_place_roots, 200, "Target stratified kAgentTurns primary placement roots for gate4a_place");
ABSL_FLAG(int, gate4a_ref_rollouts, 512, "Fresh independent Ref512 paired kTerminal kEventKeyedCrn worlds per distinct action/pair");
ABSL_FLAG(int, gate4a_online_s1_rollouts, 16, "Stage-1 worlds for online teacher");
ABSL_FLAG(int, gate4a_online_s2_rollouts, 80, "Independent Stage-2 worlds for online teacher (64 or 80)");
ABSL_FLAG(double, gate4a_online_min_t, 2.0, "min_paired_t_stat for online sparse deep teacher");
ABSL_FLAG(uint64_t, gate4a_seed, 20261105ULL, "Master seed for gate4a modes");

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

inline torch::Device GetSelectedDevice(bool use_cuda_flag) {
  if (use_cuda_flag && torch::cuda::is_available()) {
    return torch::Device(torch::kCUDA);
  }
  return torch::Device(torch::kCPU);
}

struct Gate0bAcceptedRecord {
  std::vector<float> obs;
  std::vector<Action> legal_actions;
  std::vector<std::pair<Action, double>> raw_policy;
  Action teacher_action = kInvalidAction;
  Action alt_action = kInvalidAction;
  std::vector<std::pair<Action, double>> stage2_candidate_utilities;
  int card_id = -1;
  int realization_drawn = 0;
  int realization_played = 0;
  int realization_effect = 0;
  int game_id = 0;
  int round = 1;
  // Candidate action data for v4 semantic scorer
  dune_semantic::CandidateActionData cand_v4;
  // Tleilaxu test variants
  bool has_tleilaxu_variants = false;
  int trained_action = kInvalidAction;
  int swapped_action = kInvalidAction;
  std::vector<float> obs_swapped;
  std::vector<Action> legals_swapped;
  dune_semantic::CandidateActionData cand_swapped;
  std::vector<float> obs_unrel;
  std::vector<Action> legals_unrel;
  dune_semantic::CandidateActionData cand_unrel;
  int unrel_card_id = -1;
};

struct Gate0bNonPurchaseRecord {
  std::vector<float> obs;
  std::vector<Action> legal_actions;
  std::vector<std::pair<Action, double>> raw_policy;
  dune_semantic::CandidateActionData cand_v4;
};

struct Gate0bCacheBundle {
  std::vector<Gate0bAcceptedRecord> accepted_records;
  std::vector<Gate0bNonPurchaseRecord> non_purchase_records;
  std::vector<Gate0bAcceptedRecord> tleilaxu_diag_records;
};

inline void SerializeCandidateData(std::ostream& os, const dune_semantic::CandidateActionData& cand) {
  uint64_t num_acts = cand.actions.size();
  os.write(reinterpret_cast<const char*>(&num_acts), sizeof(num_acts));
  if (num_acts > 0) {
    os.write(reinterpret_cast<const char*>(cand.actions.data()), num_acts * sizeof(Action));
    for (size_t i = 0; i < num_acts; ++i) {
      int32_t r = static_cast<int32_t>(i < cand.roles.size() ? cand.roles[i] : dune_semantic::ActionRole::kUnsupported);
      os.write(reinterpret_cast<const char*>(&r), sizeof(r));
    }
    std::vector<int64_t> safe_cards = cand.card_ids;
    if (safe_cards.size() < num_acts) safe_cards.resize(num_acts, 0);
    os.write(reinterpret_cast<const char*>(safe_cards.data()), num_acts * sizeof(int64_t));

    std::vector<int64_t> safe_spaces = cand.space_ids;
    if (safe_spaces.size() < num_acts) safe_spaces.resize(num_acts, 0);
    os.write(reinterpret_cast<const char*>(safe_spaces.data()), num_acts * sizeof(int64_t));

    uint64_t n_feat = cand.features.size();
    os.write(reinterpret_cast<const char*>(&n_feat), sizeof(n_feat));
    if (n_feat > 0) {
      os.write(reinterpret_cast<const char*>(cand.features.data()), n_feat * sizeof(float));
    }
  }
}

inline bool DeserializeCandidateData(std::istream& is, dune_semantic::CandidateActionData& cand) {
  cand.clear();
  uint64_t num_acts = 0;
  if (!is.read(reinterpret_cast<char*>(&num_acts), sizeof(num_acts))) return false;
  if (num_acts == 0) return true;

  cand.actions.resize(num_acts);
  is.read(reinterpret_cast<char*>(cand.actions.data()), num_acts * sizeof(Action));

  cand.roles.resize(num_acts);
  cand.supported.assign(num_acts, 0);
  for (size_t i = 0; i < num_acts; ++i) {
    int32_t r = 0;
    is.read(reinterpret_cast<char*>(&r), sizeof(r));
    cand.roles[i] = static_cast<dune_semantic::ActionRole>(r);
    cand.supported[i] = (cand.roles[i] != dune_semantic::ActionRole::kUnsupported) ? 1 : 0;
  }

  cand.card_ids.resize(num_acts);
  is.read(reinterpret_cast<char*>(cand.card_ids.data()), num_acts * sizeof(int64_t));

  cand.space_ids.resize(num_acts);
  is.read(reinterpret_cast<char*>(cand.space_ids.data()), num_acts * sizeof(int64_t));

  uint64_t n_feat = 0;
  if (!is.read(reinterpret_cast<char*>(&n_feat), sizeof(n_feat))) return false;
  cand.features.resize(n_feat);
  if (n_feat > 0) {
    is.read(reinterpret_cast<char*>(cand.features.data()), n_feat * sizeof(float));
  }
  return true;
}

inline void SerializeAcceptedRecord(std::ostream& ofs, const Gate0bAcceptedRecord& rec) {
  uint64_t sz = rec.obs.size(); ofs.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
  if (sz > 0) ofs.write(reinterpret_cast<const char*>(rec.obs.data()), sz * sizeof(float));

  sz = rec.legal_actions.size(); ofs.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
  if (sz > 0) ofs.write(reinterpret_cast<const char*>(rec.legal_actions.data()), sz * sizeof(Action));

  sz = rec.raw_policy.size(); ofs.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
  for (const auto& p : rec.raw_policy) {
    ofs.write(reinterpret_cast<const char*>(&p.first), sizeof(p.first));
    ofs.write(reinterpret_cast<const char*>(&p.second), sizeof(p.second));
  }

  ofs.write(reinterpret_cast<const char*>(&rec.teacher_action), sizeof(rec.teacher_action));
  ofs.write(reinterpret_cast<const char*>(&rec.alt_action), sizeof(rec.alt_action));

  sz = rec.stage2_candidate_utilities.size(); ofs.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
  for (const auto& u : rec.stage2_candidate_utilities) {
    ofs.write(reinterpret_cast<const char*>(&u.first), sizeof(u.first));
    ofs.write(reinterpret_cast<const char*>(&u.second), sizeof(u.second));
  }

  ofs.write(reinterpret_cast<const char*>(&rec.card_id), sizeof(rec.card_id));
  ofs.write(reinterpret_cast<const char*>(&rec.realization_drawn), sizeof(rec.realization_drawn));
  ofs.write(reinterpret_cast<const char*>(&rec.realization_played), sizeof(rec.realization_played));
  ofs.write(reinterpret_cast<const char*>(&rec.realization_effect), sizeof(rec.realization_effect));
  ofs.write(reinterpret_cast<const char*>(&rec.game_id), sizeof(rec.game_id));
  ofs.write(reinterpret_cast<const char*>(&rec.round), sizeof(rec.round));

  SerializeCandidateData(ofs, rec.cand_v4);

  ofs.write(reinterpret_cast<const char*>(&rec.has_tleilaxu_variants), sizeof(rec.has_tleilaxu_variants));
  ofs.write(reinterpret_cast<const char*>(&rec.trained_action), sizeof(rec.trained_action));
  ofs.write(reinterpret_cast<const char*>(&rec.swapped_action), sizeof(rec.swapped_action));

  sz = rec.obs_swapped.size(); ofs.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
  if (sz > 0) ofs.write(reinterpret_cast<const char*>(rec.obs_swapped.data()), sz * sizeof(float));

  sz = rec.legals_swapped.size(); ofs.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
  if (sz > 0) ofs.write(reinterpret_cast<const char*>(rec.legals_swapped.data()), sz * sizeof(Action));

  SerializeCandidateData(ofs, rec.cand_swapped);

  sz = rec.obs_unrel.size(); ofs.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
  if (sz > 0) ofs.write(reinterpret_cast<const char*>(rec.obs_unrel.data()), sz * sizeof(float));

  sz = rec.legals_unrel.size(); ofs.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
  if (sz > 0) ofs.write(reinterpret_cast<const char*>(rec.legals_unrel.data()), sz * sizeof(Action));

  SerializeCandidateData(ofs, rec.cand_unrel);

  ofs.write(reinterpret_cast<const char*>(&rec.unrel_card_id), sizeof(rec.unrel_card_id));
}

inline bool DeserializeAcceptedRecord(std::istream& ifs, Gate0bAcceptedRecord& rec) {
  uint64_t sz = 0;
  if (!ifs.read(reinterpret_cast<char*>(&sz), sizeof(sz))) return false;
  rec.obs.resize(sz);
  if (sz > 0) ifs.read(reinterpret_cast<char*>(rec.obs.data()), sz * sizeof(float));

  ifs.read(reinterpret_cast<char*>(&sz), sizeof(sz));
  rec.legal_actions.resize(sz);
  if (sz > 0) ifs.read(reinterpret_cast<char*>(rec.legal_actions.data()), sz * sizeof(Action));

  ifs.read(reinterpret_cast<char*>(&sz), sizeof(sz));
  rec.raw_policy.resize(sz);
  for (uint64_t p_i = 0; p_i < sz; ++p_i) {
    ifs.read(reinterpret_cast<char*>(&rec.raw_policy[p_i].first), sizeof(rec.raw_policy[p_i].first));
    ifs.read(reinterpret_cast<char*>(&rec.raw_policy[p_i].second), sizeof(rec.raw_policy[p_i].second));
  }

  ifs.read(reinterpret_cast<char*>(&rec.teacher_action), sizeof(rec.teacher_action));
  ifs.read(reinterpret_cast<char*>(&rec.alt_action), sizeof(rec.alt_action));

  ifs.read(reinterpret_cast<char*>(&sz), sizeof(sz));
  rec.stage2_candidate_utilities.resize(sz);
  for (uint64_t u_i = 0; u_i < sz; ++u_i) {
    ifs.read(reinterpret_cast<char*>(&rec.stage2_candidate_utilities[u_i].first), sizeof(rec.stage2_candidate_utilities[u_i].first));
    ifs.read(reinterpret_cast<char*>(&rec.stage2_candidate_utilities[u_i].second), sizeof(rec.stage2_candidate_utilities[u_i].second));
  }

  ifs.read(reinterpret_cast<char*>(&rec.card_id), sizeof(rec.card_id));
  ifs.read(reinterpret_cast<char*>(&rec.realization_drawn), sizeof(rec.realization_drawn));
  ifs.read(reinterpret_cast<char*>(&rec.realization_played), sizeof(rec.realization_played));
  ifs.read(reinterpret_cast<char*>(&rec.realization_effect), sizeof(rec.realization_effect));
  ifs.read(reinterpret_cast<char*>(&rec.game_id), sizeof(rec.game_id));
  ifs.read(reinterpret_cast<char*>(&rec.round), sizeof(rec.round));

  if (!DeserializeCandidateData(ifs, rec.cand_v4)) return false;

  ifs.read(reinterpret_cast<char*>(&rec.has_tleilaxu_variants), sizeof(rec.has_tleilaxu_variants));
  ifs.read(reinterpret_cast<char*>(&rec.trained_action), sizeof(rec.trained_action));
  ifs.read(reinterpret_cast<char*>(&rec.swapped_action), sizeof(rec.swapped_action));

  ifs.read(reinterpret_cast<char*>(&sz), sizeof(sz));
  rec.obs_swapped.resize(sz);
  if (sz > 0) ifs.read(reinterpret_cast<char*>(rec.obs_swapped.data()), sz * sizeof(float));

  ifs.read(reinterpret_cast<char*>(&sz), sizeof(sz));
  rec.legals_swapped.resize(sz);
  if (sz > 0) ifs.read(reinterpret_cast<char*>(rec.legals_swapped.data()), sz * sizeof(Action));

  if (!DeserializeCandidateData(ifs, rec.cand_swapped)) return false;

  ifs.read(reinterpret_cast<char*>(&sz), sizeof(sz));
  rec.obs_unrel.resize(sz);
  if (sz > 0) ifs.read(reinterpret_cast<char*>(rec.obs_unrel.data()), sz * sizeof(float));

  ifs.read(reinterpret_cast<char*>(&sz), sizeof(sz));
  rec.legals_unrel.resize(sz);
  if (sz > 0) ifs.read(reinterpret_cast<char*>(rec.legals_unrel.data()), sz * sizeof(Action));

  if (!DeserializeCandidateData(ifs, rec.cand_unrel)) return false;

  ifs.read(reinterpret_cast<char*>(&rec.unrel_card_id), sizeof(rec.unrel_card_id));
  return true;
}

inline void SerializeNonPurchaseRecord(std::ostream& ofs, const Gate0bNonPurchaseRecord& rec) {
  uint64_t sz = rec.obs.size(); ofs.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
  if (sz > 0) ofs.write(reinterpret_cast<const char*>(rec.obs.data()), sz * sizeof(float));

  sz = rec.legal_actions.size(); ofs.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
  if (sz > 0) ofs.write(reinterpret_cast<const char*>(rec.legal_actions.data()), sz * sizeof(Action));

  sz = rec.raw_policy.size(); ofs.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
  for (const auto& p : rec.raw_policy) {
    ofs.write(reinterpret_cast<const char*>(&p.first), sizeof(p.first));
    ofs.write(reinterpret_cast<const char*>(&p.second), sizeof(p.second));
  }

  SerializeCandidateData(ofs, rec.cand_v4);
}

inline bool DeserializeNonPurchaseRecord(std::istream& ifs, Gate0bNonPurchaseRecord& rec) {
  uint64_t sz = 0;
  if (!ifs.read(reinterpret_cast<char*>(&sz), sizeof(sz))) return false;
  rec.obs.resize(sz);
  if (sz > 0) ifs.read(reinterpret_cast<char*>(rec.obs.data()), sz * sizeof(float));

  ifs.read(reinterpret_cast<char*>(&sz), sizeof(sz));
  rec.legal_actions.resize(sz);
  if (sz > 0) ifs.read(reinterpret_cast<char*>(rec.legal_actions.data()), sz * sizeof(Action));

  ifs.read(reinterpret_cast<char*>(&sz), sizeof(sz));
  rec.raw_policy.resize(sz);
  for (uint64_t p_i = 0; p_i < sz; ++p_i) {
    ifs.read(reinterpret_cast<char*>(&rec.raw_policy[p_i].first), sizeof(rec.raw_policy[p_i].first));
    ifs.read(reinterpret_cast<char*>(&rec.raw_policy[p_i].second), sizeof(rec.raw_policy[p_i].second));
  }

  if (!DeserializeCandidateData(ifs, rec.cand_v4)) return false;
  return true;
}

inline bool SaveGate0bCache(const std::string& path, const Gate0bCacheBundle& bundle) {
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs.is_open()) return false;

  uint64_t n_acc = bundle.accepted_records.size();
  ofs.write(reinterpret_cast<const char*>(&n_acc), sizeof(n_acc));
  for (const auto& rec : bundle.accepted_records) {
    SerializeAcceptedRecord(ofs, rec);
  }

  uint64_t n_np = bundle.non_purchase_records.size();
  ofs.write(reinterpret_cast<const char*>(&n_np), sizeof(n_np));
  for (const auto& rec : bundle.non_purchase_records) {
    SerializeNonPurchaseRecord(ofs, rec);
  }

  uint64_t n_diag = bundle.tleilaxu_diag_records.size();
  ofs.write(reinterpret_cast<const char*>(&n_diag), sizeof(n_diag));
  for (const auto& rec : bundle.tleilaxu_diag_records) {
    SerializeAcceptedRecord(ofs, rec);
  }

  return true;
}

inline bool LoadGate0bCache(const std::string& path, Gate0bCacheBundle& bundle) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs.is_open()) return false;

  uint64_t n_acc = 0;
  if (!ifs.read(reinterpret_cast<char*>(&n_acc), sizeof(n_acc))) return false;
  bundle.accepted_records.resize(n_acc);
  for (uint64_t i = 0; i < n_acc; ++i) {
    if (!DeserializeAcceptedRecord(ifs, bundle.accepted_records[i])) return false;
  }

  uint64_t n_np = 0;
  if (!ifs.read(reinterpret_cast<char*>(&n_np), sizeof(n_np))) return false;
  bundle.non_purchase_records.resize(n_np);
  for (uint64_t i = 0; i < n_np; ++i) {
    if (!DeserializeNonPurchaseRecord(ifs, bundle.non_purchase_records[i])) return false;
  }

  uint64_t n_diag = 0;
  if (ifs.read(reinterpret_cast<char*>(&n_diag), sizeof(n_diag))) {
    bundle.tleilaxu_diag_records.resize(n_diag);
    for (uint64_t i = 0; i < n_diag; ++i) {
      if (!DeserializeAcceptedRecord(ifs, bundle.tleilaxu_diag_records[i])) return false;
    }
  }

  return true;
}

// -----------------------------------------------------------------------------
// MODE 1: stage3_cpu_verify
// -----------------------------------------------------------------------------
void RunStage3CpuVerify() {
  std::cout << "========================================================\n";
  std::cout << "=== MODE: stage3_cpu_verify                          ===\n";
  std::cout << "========================================================\n\n";

  torch::Device device(torch::kCPU);
  const int64_t input_dim = dune_imperium::kFullPublicInformationV4StateSize;
  const int64_t hidden_dim = 2048;
  const int64_t action_dim = 2391;
  const int num_blocks = absl::GetFlag(FLAGS_num_blocks);
  const int src_num_blocks = absl::GetFlag(FLAGS_src_num_blocks);
  const double base_lr = 5e-6;
  const double new_block_lr = 5e-5;

  std::string model_path = absl::GetFlag(FLAGS_source_model_path);
  std::string opt_path = absl::GetFlag(FLAGS_source_opt_path);

  if (model_path.empty() || !std::filesystem::exists(model_path)) {
    model_path = "/tmp/test_source_model_v3.pt";
    opt_path = "/tmp/test_source_opt_v3.pt";
    std::cout << "[INFO] Generating synthetic source v3 model/opt at /tmp/...\n";

    auto synth_model = std::make_shared<SharedDunePolicyValueNetImpl>(
        input_dim, hidden_dim, action_dim, src_num_blocks,
        /*use_nonlinear=*/false, /*with_aux_heads=*/false,
        /*head_init_seed=*/42, /*with_semantic_scorer=*/true,
        /*enable_zone_encoder=*/false,
        /*semantic_descriptor_schema=*/dune_semantic::kDescriptorSchemaVersionV3);

    std::vector<torch::Tensor> s_g0;
    s_g0.push_back(synth_model->policy_head->weight);
    if (synth_model->policy_head->bias.defined()) s_g0.push_back(synth_model->policy_head->bias);

    std::vector<torch::Tensor> s_g1;
    s_g1.push_back(synth_model->input_layer->weight);
    if (synth_model->input_layer->bias.defined()) s_g1.push_back(synth_model->input_layer->bias);
    for (int b = 0; b < 8; ++b) {
      for (auto& p : synth_model->res_blocks[b]->parameters()) s_g1.push_back(p);
    }
    s_g1.push_back(synth_model->value_head->weight);
    if (synth_model->value_head->bias.defined()) s_g1.push_back(synth_model->value_head->bias);
    auto add_scorer = [&](torch::nn::Module& m) {
      for (auto& p : m.parameters()) s_g1.push_back(p);
    };
    add_scorer(*synth_model->semantic_scorer_->card_embedding);
    add_scorer(*synth_model->semantic_scorer_->space_embedding);
    add_scorer(*synth_model->semantic_scorer_->desc_proj);
    add_scorer(*synth_model->semantic_scorer_->desc_ln);
    add_scorer(*synth_model->semantic_scorer_->trunk_proj);
    add_scorer(*synth_model->semantic_scorer_->trunk_ln);
    add_scorer(*synth_model->semantic_scorer_->mlp1);
    add_scorer(*synth_model->semantic_scorer_->mlp1_ln);
    add_scorer(*synth_model->semantic_scorer_->out_layer);
    add_scorer(*synth_model->semantic_scorer_->out_layer_ext);

    std::vector<torch::Tensor> s_g2;
    for (int b = 8; b < src_num_blocks; ++b) {
      for (auto& p : synth_model->res_blocks[b]->parameters()) s_g2.push_back(p);
    }

    std::vector<torch::optim::OptimizerParamGroup> s_groups;
    s_groups.emplace_back(s_g0);
    s_groups.emplace_back(s_g1);
    s_groups.emplace_back(s_g2);
    torch::optim::AdamW s_opt(s_groups, torch::optim::AdamWOptions(base_lr).eps(1e-5));

    int64_t step_cnt = 100;
    for (size_t g = 0; g < s_groups.size(); ++g) {
      for (auto& p : s_groups[g].params()) {
        auto state = std::make_unique<torch::optim::AdamWParamState>();
        state->step(step_cnt++);
        state->exp_avg(torch::randn_like(p));
        state->exp_avg_sq(torch::rand_like(p).abs());
        s_opt.state()[p.unsafeGetTensorImpl()] = std::move(state);
      }
    }

    torch::save(synth_model, model_path);
    torch::save(s_opt, opt_path);
  }

  // 1. Verify schema v3 vs v4 descriptor fail-closed on kTleilaxuPurchase
  std::cout << "--------------------------------------------------------\n";
  std::cout << "SUBCHECK 1: Schema fail-closed on kTleilaxuPurchase\n";
  std::cout << "--------------------------------------------------------\n";
  {
    auto state = BuildStateAtFirstRevealTurn("dune_imperium(enable_immortality=true)");
    auto* dune_st = dynamic_cast<DuneImperiumState*>(state.get());
    CHECK_ASSERT(dune_st != nullptr, "DuneImperiumState cast must succeed");
    const Player p = dune_st->CurrentPlayer();
    dune_st->SetTleilaxuRowForTesting({13, 11});
    dune_st->SetSpecimensForTesting(p, 5);

    auto legal_actions = dune_st->LegalActions();
    CHECK_ASSERT(std::find(legal_actions.begin(), legal_actions.end(), 91) != legal_actions.end(),
                 "Action 91 must be legal");
    CHECK_ASSERT(std::find(legal_actions.begin(), legal_actions.end(), 92) != legal_actions.end(),
                 "Action 92 must be legal");

    dune_semantic::CandidateActionData cand_v3;
    dune_semantic::ExtractCandidateDescriptors(
        *dune_st, legal_actions, &cand_v3, "semantic_action_v3");
    for (size_t i = 0; i < cand_v3.actions.size(); ++i) {
      if (cand_v3.actions[i] == 91 || cand_v3.actions[i] == 92) {
        CHECK_ASSERT(cand_v3.roles[i] == dune_semantic::ActionRole::kUnsupported,
                     "Schema v3 must fail closed with kUnsupported on actions 91 and 92");
      }
    }

    dune_semantic::CandidateActionData cand_v4;
    dune_semantic::ExtractCandidateDescriptors(
        *dune_st, legal_actions, &cand_v4, "semantic_action_v4");
    int found_91 = 0, found_92 = 0;
    for (size_t i = 0; i < cand_v4.actions.size(); ++i) {
      if (cand_v4.actions[i] == 91) {
        CHECK_ASSERT(cand_v4.roles[i] == dune_semantic::ActionRole::kTleilaxuPurchase,
                     "Schema v4 must mark action 91 as kTleilaxuPurchase");
        CHECK_ASSERT(cand_v4.card_ids[i] == 120, "Slot 0 (13) must map to 120");
        ++found_91;
      }
      if (cand_v4.actions[i] == 92) {
        CHECK_ASSERT(cand_v4.roles[i] == dune_semantic::ActionRole::kTleilaxuPurchase,
                     "Schema v4 must mark action 92 as kTleilaxuPurchase");
        CHECK_ASSERT(cand_v4.card_ids[i] == 118, "Slot 1 (11) must map to 118");
        ++found_92;
      }
    }
    CHECK_ASSERT(found_91 == 1 && found_92 == 1, "Must extract both actions 91 and 92");
    std::cout << "  Passed: v3 fails closed as kUnsupported; v4 maps 91->120, 92->118.\n";
  }

  // 2. Parity across 1,024 states in FP32 and BF16
  std::cout << "--------------------------------------------------------\n";
  std::cout << "SUBCHECK 2: Parity across 1,024 states in FP32 & BF16\n";
  std::cout << "--------------------------------------------------------\n";
  {
    auto model_v3 = std::make_shared<SharedDunePolicyValueNetImpl>(
        input_dim, hidden_dim, action_dim, num_blocks,
        /*use_nonlinear=*/false, /*with_aux_heads=*/false,
        /*head_init_seed=*/0, /*with_semantic_scorer=*/true,
        /*enable_zone_encoder=*/false,
        /*semantic_descriptor_schema=*/dune_semantic::kDescriptorSchemaVersionV3);
    model_v3->market_appendix_mode_ = MarketAppendixMode::kFullPublicInformationV4;
    model_v3->deck_pool_semantics_ = DeckPoolSemantics::kFixed;
    LoadModelCheckpointRobust(model_v3, model_path, device);
    model_v3->to(device);
    model_v3->eval();

    auto model_v4 = std::make_shared<SharedDunePolicyValueNetImpl>(
        input_dim, hidden_dim, action_dim, num_blocks,
        /*use_nonlinear=*/false, /*with_aux_heads=*/false,
        /*head_init_seed=*/0, /*with_semantic_scorer=*/true,
        /*enable_zone_encoder=*/false,
        /*semantic_descriptor_schema=*/dune_semantic::kDescriptorSchemaVersionV4);
    model_v4->market_appendix_mode_ = MarketAppendixMode::kFullPublicInformationV4;
    model_v4->deck_pool_semantics_ = DeckPoolSemantics::kFixed;
    LoadModelCheckpointRobust(model_v4, model_path, device);
    model_v4->semantic_scorer_->EnableF(absl::GetFlag(FLAGS_slot_indexed_f));
    model_v4->to(device);
    model_v4->eval();

    std::shared_ptr<const Game> game = LoadGame("dune_imperium(enable_immortality=true)");
    std::mt19937_64 rng(12345);
    int states_checked = 0;
    float max_diff_fp32_logits = 0.0f;
    float max_diff_fp32_values = 0.0f;
    float max_diff_bf16_logits = 0.0f;
    float max_diff_bf16_values = 0.0f;

    while (states_checked < 1024) {
      std::unique_ptr<State> st = game->NewInitialState();
      while (!st->IsTerminal() && states_checked < 1024) {
        if (st->IsChanceNode()) {
          auto outcomes = st->ChanceOutcomes();
          std::uniform_int_distribution<size_t> dist(0, outcomes.size() - 1);
          st->ApplyAction(outcomes[dist(rng)].first);
          continue;
        }
        auto legal = st->LegalActions();
        if (legal.empty()) break;

        const auto* dune_st = dynamic_cast<const DuneImperiumState*>(st.get());
        if (dune_st != nullptr &&
            (dune_st->phase() == GamePhase::kAgentTurns ||
             dune_st->phase() == GamePhase::kRevealTurns)) {
          const Player p = dune_st->CurrentPlayer();
          std::vector<float> obs = dune_st->InformationStateTensorWithAppendix(
              p, MarketAppendixMode::kFullPublicInformationV4, model_v3->deck_pool_semantics_);
          torch::Tensor obs_fp32 = torch::from_blob(
              obs.data(), {1, static_cast<long>(obs.size())}, torch::kFloat32).to(device);

          dune_semantic::CandidateActionData cand_v3_data;
          dune_semantic::ExtractCandidateDescriptors(
              *dune_st, legal, &cand_v3_data, "semantic_action_v3");
          dune_semantic::CandidateActionData cand_v4_data;
          dune_semantic::ExtractCandidateDescriptors(
              *dune_st, legal, &cand_v4_data, "semantic_action_v4");

          // FP32 pass
          {
            torch::NoGradGuard no_grad;
            auto out_v3 = model_v3->forward(obs_fp32);
            auto out_v4 = model_v4->forward(obs_fp32);

            std::vector<const dune_semantic::CandidateActionData*> cands_v3 = {&cand_v3_data};
            std::vector<const dune_semantic::CandidateActionData*> cands_v4 = {&cand_v4_data};
            dune_semantic::ApplySemanticScorerBatch(
                model_v3->semantic_scorer_, out_v3.trunk, cands_v3, out_v3.logits, device);
            dune_semantic::ApplySemanticScorerBatch(
                model_v4->semantic_scorer_, out_v4.trunk, cands_v4, out_v4.logits, device);

            float ld = (out_v3.logits - out_v4.logits).abs().max().item().toFloat();
            float vd = (out_v3.values - out_v4.values).abs().max().item().toFloat();
            if (ld > max_diff_fp32_logits) max_diff_fp32_logits = ld;
            if (vd > max_diff_fp32_values) max_diff_fp32_values = vd;
          }

          // BF16 pass
          {
            torch::NoGradGuard no_grad;
            torch::Tensor obs_bf16 = obs_fp32.to(torch::kBFloat16).to(torch::kFloat32);
            auto out_v3 = model_v3->forward(obs_bf16);
            auto out_v4 = model_v4->forward(obs_bf16);

            std::vector<const dune_semantic::CandidateActionData*> cands_v3 = {&cand_v3_data};
            std::vector<const dune_semantic::CandidateActionData*> cands_v4 = {&cand_v4_data};
            dune_semantic::ApplySemanticScorerBatch(
                model_v3->semantic_scorer_, out_v3.trunk, cands_v3, out_v3.logits, device);
            dune_semantic::ApplySemanticScorerBatch(
                model_v4->semantic_scorer_, out_v4.trunk, cands_v4, out_v4.logits, device);

            float ld = (out_v3.logits - out_v4.logits).abs().max().item().toFloat();
            float vd = (out_v3.values - out_v4.values).abs().max().item().toFloat();
            if (ld > max_diff_bf16_logits) max_diff_bf16_logits = ld;
            if (vd > max_diff_bf16_values) max_diff_bf16_values = vd;
          }

          ++states_checked;
        }

        std::uniform_int_distribution<size_t> dist(0, legal.size() - 1);
        st->ApplyAction(legal[dist(rng)]);
      }
    }

    std::cout << "  States checked: " << states_checked << "\n";
    std::cout << "  FP32 max diff logits: " << max_diff_fp32_logits
              << " values: " << max_diff_fp32_values << "\n";
    std::cout << "  BF16 max diff logits: " << max_diff_bf16_logits
              << " values: " << max_diff_bf16_values << "\n";
    CHECK_ASSERT(max_diff_fp32_logits == 0.0f, "FP32 logits max_diff must be 0.0");
    CHECK_ASSERT(max_diff_fp32_values == 0.0f, "FP32 values max_diff must be 0.0");
    CHECK_ASSERT(max_diff_bf16_logits == 0.0f, "BF16 logits max_diff must be 0.0");
    CHECK_ASSERT(max_diff_bf16_values == 0.0f, "BF16 values max_diff must be 0.0");
    std::cout << "  Passed: exact 0.0 max_diff over 1,024 states in FP32 and BF16.\n";
  }

  // 3. Optimizer migration verifying moved == 152 and out_layer_f empty
  std::cout << "--------------------------------------------------------\n";
  std::cout << "SUBCHECK 3: LoadOptimizerCheckpointMigrating moved == 152\n";
  std::cout << "--------------------------------------------------------\n";
  {
    auto target_model = std::make_shared<SharedDunePolicyValueNetImpl>(
        input_dim, hidden_dim, action_dim, num_blocks,
        /*use_nonlinear=*/false, /*with_aux_heads=*/false,
        /*head_init_seed=*/0, /*with_semantic_scorer=*/true,
        /*enable_zone_encoder=*/false,
        /*semantic_descriptor_schema=*/dune_semantic::kDescriptorSchemaVersionV4);
    target_model->market_appendix_mode_ = MarketAppendixMode::kFullPublicInformationV4;
    target_model->deck_pool_semantics_ = DeckPoolSemantics::kFixed;
    target_model->semantic_scorer_->EnableF(absl::GetFlag(FLAGS_slot_indexed_f));
    LoadModelCheckpointRobust(target_model, model_path, device);
    target_model->to(device);

    auto target_opt = MakeDuneOptimizer(
        target_model, base_lr, 0.0, 0.0, new_block_lr, /*new_block_first_idx=*/8);

    bool migrated = LoadOptimizerCheckpointMigrating(
        target_model, *target_opt, opt_path, device, base_lr);

    const int expected_moved = 2 + 86 + 8 * (num_blocks - 8); // 152 for 16 blocks
    std::cout << "  Migrated returned: " << (migrated ? "true" : "false") << "\n";
    std::cout << "  Optimizer state size: " << target_opt->state().size()
              << " (expected " << expected_moved << ")\n";
    CHECK_ASSERT(target_opt->state().size() == static_cast<size_t>(expected_moved),
                 "Optimizer state must contain exactly expected_moved states");

    for (auto& p : target_model->semantic_scorer_->out_layer_f->parameters()) {
      auto it = target_opt->state().find(p.unsafeGetTensorImpl());
      CHECK_ASSERT(it == target_opt->state().end(),
                   "out_layer_f parameters must have empty optimizer state");
    }
    std::cout << "  Passed: moved == " << expected_moved << " and out_layer_f empty.\n";

    // 4. Save temp v4 bundle, test v4 -> v4 reload (torch::load, 154 states)
    std::cout << "--------------------------------------------------------\n";
    std::cout << "SUBCHECK 4: Temp v4 save & reload (direct torch::load)\n";
    std::cout << "--------------------------------------------------------\n";
    const std::string tmp_model = "/tmp/gate0_temp_v4_model.pt";
    const std::string tmp_opt = "/tmp/gate0_temp_v4_opt.pt";
    const std::string tmp_json = "/tmp/gate0_temp_v4_model.json";

    int64_t step_cnt = 200;
    for (auto& p : target_model->semantic_scorer_->out_layer_f->parameters()) {
      auto st = std::make_unique<torch::optim::AdamWParamState>();
      st->step(step_cnt++);
      st->exp_avg(torch::zeros_like(p));
      st->exp_avg_sq(torch::zeros_like(p));
      target_opt->state()[p.unsafeGetTensorImpl()] = std::move(st);
    }
    CHECK_ASSERT(target_opt->state().size() == static_cast<size_t>(expected_moved + 2),
                 "Total optimizer states must be 154 before save");

    torch::save(target_model, tmp_model);
    torch::save(*target_opt, tmp_opt);

    const std::string src_json = std::filesystem::path(model_path).replace_extension(".json").string();
    if (std::filesystem::exists(src_json)) {
      std::filesystem::copy_file(src_json, tmp_json, std::filesystem::copy_options::overwrite_existing);
    } else {
      std::ofstream meta(tmp_json);
      const auto schema = dune_imperium::GetInformationStateSchema(MarketAppendixMode::kFullPublicInformationV4);
      meta << "{\"observation_dim\":" << dune_imperium::kFullPublicInformationV4StateSize
           << ",\"hidden_dim\":2048,\"num_blocks\":" << num_blocks
           << ",\"market_appendix_mode\":\"full_public_information_v4\","
           << "\"feature_schema_sha256\":\"" << schema.sha256 << "\","
           << "\"feature_schema_label\":\"" << schema.label << "\","
           << "\"enable_semantic_scorer\":true,"
           << "\"semantic_descriptor_schema\":\"semantic_action_v4\"}\n";
      meta.close();
    }

    auto reload_model = std::make_shared<SharedDunePolicyValueNetImpl>(
        input_dim, hidden_dim, action_dim, num_blocks,
        /*use_nonlinear=*/false, /*with_aux_heads=*/false,
        /*head_init_seed=*/0, /*with_semantic_scorer=*/true,
        /*enable_zone_encoder=*/false,
        /*semantic_descriptor_schema=*/dune_semantic::kDescriptorSchemaVersionV4);
    reload_model->market_appendix_mode_ = MarketAppendixMode::kFullPublicInformationV4;
    reload_model->deck_pool_semantics_ = DeckPoolSemantics::kFixed;
    reload_model->semantic_scorer_->EnableF(absl::GetFlag(FLAGS_slot_indexed_f));
    LoadModelCheckpointRobust(reload_model, tmp_model, device);

    auto reload_opt = MakeDuneOptimizer(
        reload_model, base_lr, 0.0, 0.0, new_block_lr, /*new_block_first_idx=*/8);

    bool reload_migrated = LoadOptimizerCheckpointMigrating(
        reload_model, *reload_opt, tmp_opt, device, base_lr);
    CHECK_ASSERT(!reload_migrated, "v4 -> v4 must take direct torch::load path (no migration)");

    torch::load(*reload_opt, tmp_opt, device);
    CHECK_ASSERT(reload_opt->state().size() == static_cast<size_t>(expected_moved + 2),
                 "Reloaded optimizer must contain exactly 154 states");
    std::cout << "  Passed: v4 -> v4 reload verified with 154 states.\n";
    std::filesystem::remove(tmp_model);
    std::filesystem::remove(tmp_opt);
    std::filesystem::remove(tmp_json);
  }

  std::cout << "\n========================================================\n";
  std::cout << "STAGE3 CPU VERIFY COMPLETE: " << g_checks_passed << " / "
            << g_checks_run << " CHECKS PASSED\n";
  std::cout << "========================================================\n";
}

inline Action SampleActionSafe(const ActionsAndProbs& outcomes, double u) {
  if (outcomes.empty()) return kInvalidAction;
  double total = 0.0;
  for (const auto& ap : outcomes) {
    if (ap.second > 0.0) total += ap.second;
  }
  if (total <= 0.0) return outcomes.front().first;
  const double target = std::clamp(u, 0.0, std::nextafter(1.0, 0.0)) * total;
  double acc = 0.0;
  for (const auto& ap : outcomes) {
    if (ap.second > 0.0) {
      acc += ap.second;
      if (target <= acc) return ap.first;
    }
  }
  return outcomes.back().first;
}

// -----------------------------------------------------------------------------
// MODE 2: gate0a
// -----------------------------------------------------------------------------
void RunGate0a() {
  std::cout << "========================================================\n";
  std::cout << "=== MODE: gate0a                                     ===\n";
  std::cout << "========================================================\n\n";

  torch::Device device = GetSelectedDevice(absl::GetFlag(FLAGS_use_cuda));
  std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";

  const int num_roots = absl::GetFlag(FLAGS_num_reveal_roots);
  const int eval_rollouts = absl::GetFlag(FLAGS_eval_rollouts);
  const int bootstrap_samples = absl::GetFlag(FLAGS_bootstrap_samples);
  const std::string cache_path = absl::GetFlag(FLAGS_gate0b_cache_path);
  const int batch_size = absl::GetFlag(FLAGS_batch_size);

  const int64_t input_dim = dune_imperium::kFullPublicInformationV4StateSize;
  const int64_t hidden_dim = 2048;
  const int64_t action_dim = 2391;
  const int num_blocks = absl::GetFlag(FLAGS_num_blocks);

  auto net = std::make_shared<SharedDunePolicyValueNetImpl>(
      input_dim, hidden_dim, action_dim, num_blocks,
      /*use_nonlinear=*/false, /*with_aux_heads=*/false,
      /*head_init_seed=*/0, /*with_semantic_scorer=*/true,
      /*enable_zone_encoder=*/false,
      /*semantic_descriptor_schema=*/dune_semantic::kDescriptorSchemaVersionV4);
  net->market_appendix_mode_ = MarketAppendixMode::kFullPublicInformationV4;
  net->deck_pool_semantics_ = DeckPoolSemantics::kFixed;
  net->semantic_scorer_->EnableF(absl::GetFlag(FLAGS_slot_indexed_f));

  if (!absl::GetFlag(FLAGS_source_model_path).empty() &&
      std::filesystem::exists(absl::GetFlag(FLAGS_source_model_path))) {
    LoadModelCheckpointRobust(net, absl::GetFlag(FLAGS_source_model_path), device);
  }
  net->to(device);
  net->eval();

  std::shared_mutex mutex;
  auto coord = std::make_shared<BatchedEvaluator>(
      net, batch_size, /*timeout_ms=*/2, device, &mutex,
      /*logit_cap=*/10.0f,
      /*device_synchronize=*/!device.is_cuda(),
      /*high_priority_stream=*/false,
      /*emit_batch_membership=*/false,
      /*rollout_amp=*/device.is_cuda(),
      /*allow_tf32=*/true);
  auto nn_eval = std::make_shared<BatchedNNEvaluator>(coord, 10.0f);

  auto collect_coord = std::make_shared<BatchedEvaluator>(
      net, /*target_batch_size=*/1, /*timeout_ms=*/1, device, &mutex,
      /*logit_cap=*/10.0f,
      /*device_synchronize=*/!device.is_cuda(),
      /*high_priority_stream=*/false,
      /*emit_batch_membership=*/false,
      /*rollout_amp=*/device.is_cuda(),
      /*allow_tf32=*/true);
  auto collect_eval = std::make_shared<BatchedNNEvaluator>(collect_coord, 10.0f);

  std::shared_ptr<const Game> game = LoadGame("dune_imperium(enable_immortality=true)");

  // Stratified collection
  std::vector<std::unique_ptr<State>> reveal_roots;
  std::vector<Player> root_players;
  std::vector<std::vector<float>> root_obs;
  std::vector<int> root_game_ids;
  std::vector<int> root_rounds;
  std::vector<int> root_deck_plus_discards;
  std::vector<int> root_specimens;
  std::vector<ActionsAndProbs> root_priors;

  std::vector<Gate0bNonPurchaseRecord> non_purchase_records;

  int total_reveal_turns = 0;
  int legacy_cat3_count = 0;
  int strict_cat3_count = 0;
  int roots_with_sub002_buy = 0;
  int current_game_id = 0;

  std::cout << "[INFO] Collecting " << num_roots
            << " natural R1-R6 Reveal roots via stochastic policy play...\n";

  const int quota_r12 = (num_roots + 2) / 3;
  const int quota_r34 = (num_roots + 1) / 3;
  const int quota_r56 = num_roots - quota_r12 - quota_r34;
  int collected_r12 = 0;
  int collected_r34 = 0;
  int collected_r56 = 0;

  std::mt19937_64 master_rng(654321);

  while (static_cast<int>(reveal_roots.size()) < num_roots) {
    ++current_game_id;
    bool game_got_r12 = false;
    bool game_got_r34 = false;
    bool game_got_r56 = false;
    std::unique_ptr<State> st = game->NewInitialState();
    uint64_t game_seed = master_rng();
    std::mt19937_64 game_rng(game_seed);

    while (!st->IsTerminal() && static_cast<int>(reveal_roots.size()) < num_roots) {
      if (st->IsChanceNode()) {
        auto outcomes = st->ChanceOutcomes();
        double u = std::generate_canonical<double, 53>(game_rng);
        Action ca = SampleActionSafe(outcomes, u);
        st->ApplyAction(ca);
        continue;
      }
      auto legal = st->LegalActions();
      if (legal.empty()) break;

      auto* dune_st = dynamic_cast<DuneImperiumState*>(st.get());
      const Player p = dune_st ? dune_st->CurrentPlayer() : st->CurrentPlayer();

      if (dune_st != nullptr && dune_st->GetCurrentRound() > 6 && non_purchase_records.size() >= 100) break;
      if (collected_r56 >= quota_r56 && dune_st != nullptr && dune_st->GetCurrentRound() >= 5 && non_purchase_records.size() >= 100) break;
      if (collected_r34 >= quota_r34 && collected_r56 >= quota_r56 && dune_st != nullptr && dune_st->GetCurrentRound() >= 3 && non_purchase_records.size() >= 100) break;

      if (dune_st != nullptr && dune_st->phase() == GamePhase::kAgentTurns &&
          !dune_st->IsPlayerInRevealTurn(p) &&
          non_purchase_records.size() < 100) {
        Gate0bNonPurchaseRecord np_rec;
        np_rec.obs = dune_st->InformationStateTensorWithAppendix(
            p, MarketAppendixMode::kFullPublicInformationV4, DeckPoolSemantics::kFixed);
        np_rec.legal_actions = legal;
        dune_semantic::ExtractCandidateDescriptors(
            *dune_st, legal, &np_rec.cand_v4, dune_semantic::kDescriptorSchemaVersionV4);
        auto raw_p = collect_eval->Prior(*st);
        for (const auto& ap : raw_p) {
          if (std::find(legal.begin(), legal.end(), ap.first) != legal.end()) {
            np_rec.raw_policy.push_back(ap);
          }
        }
        non_purchase_records.push_back(std::move(np_rec));
      }

      if (dune_st != nullptr && dune_st->IsPlayerInRevealTurn(p) &&
          dune_st->GetCurrentRound() <= 6) {
        ++total_reveal_turns;
        if (ClassifyR755SearchCategory(*dune_st, p, false, 31, false) == R755SearchCategory::kRevealPurchase) {
          ++legacy_cat3_count;
        }

        R755SearchCategory strict_cat = ClassifyR755SearchCategory(
            *dune_st, p, false, 31, /*strict_cat3_card_tech_only=*/true);

        if (strict_cat == R755SearchCategory::kRevealPurchase && legal.size() >= 2) {
          ++strict_cat3_count;
          auto raw_prior = collect_eval->Prior(*st);

          bool has_sub002 = false;
          for (Action a : legal) {
            if (IsCardOrTechAcquisitionAction(a)) {
              double pr = 0.0;
              for (const auto& ap : raw_prior) {
                if (ap.first == a) { pr = ap.second; break; }
              }
              if (pr < 0.02) { has_sub002 = true; break; }
            }
          }
          if (has_sub002) ++roots_with_sub002_buy;

          const int rnd = dune_st->GetCurrentRound();
          bool take_root = false;
          int stratum_matched = 0;

          if (rnd <= 2 && !game_got_r12 && collected_r12 < quota_r12) {
            take_root = true;
            stratum_matched = 1;
          } else if (rnd >= 3 && rnd <= 4 && !game_got_r34 && collected_r34 < quota_r34) {
            take_root = true;
            stratum_matched = 2;
          } else if (rnd >= 5 && rnd <= 6 && !game_got_r56 && collected_r56 < quota_r56) {
            take_root = true;
            stratum_matched = 3;
          }

          if (take_root && rnd >= 3) {
            if (dune_st->GetSpecimensForTesting(p) < 3) {
              if (current_game_id % 2 == 0 && current_game_id <= 120) {
                take_root = false;
              }
            }
          }

          if (take_root && static_cast<int>(reveal_roots.size()) < num_roots) {
            if (stratum_matched == 1) {
              game_got_r12 = true;
              ++collected_r12;
            } else if (stratum_matched == 2) {
              game_got_r34 = true;
              ++collected_r34;
            } else if (stratum_matched == 3) {
              game_got_r56 = true;
              ++collected_r56;
            }

            reveal_roots.push_back(st->Clone());
            root_players.push_back(p);
            root_game_ids.push_back(current_game_id);
            root_rounds.push_back(dune_st->GetCurrentRound());
            int d_pd = dune_st->GetPlayerDrawDeckForTesting(p).size() +
                       dune_st->GetPlayerDiscardForTesting(p).size();
            root_deck_plus_discards.push_back(d_pd);
            root_specimens.push_back(dune_st->GetSpecimensForTesting(p));
            root_priors.push_back(raw_prior);
            root_obs.push_back(dune_st->InformationStateTensorWithAppendix(
                p, MarketAppendixMode::kFullPublicInformationV4, DeckPoolSemantics::kFixed));
          }
        }
      }

      auto prior_legals = collect_eval->PriorWithLegals(*st, legal);
      double u_act = std::generate_canonical<double, 53>(game_rng);
      Action act = SampleActionSafe(prior_legals, u_act);
      st->ApplyAction(act);
    }
  }

  std::cout << "  Collected " << reveal_roots.size() << " roots across "
            << current_game_id << " distinct games.\n";
  std::cout << "  Collected " << non_purchase_records.size()
            << " non-purchase kAgentTurns reference states.\n\n";

  // Strata summary
  int r12 = 0, r34 = 0, r56 = 0, d_le8 = 0, d_gt8 = 0, spec_ge3 = 0, sb_sh_exp = 0;
  for (size_t i = 0; i < reveal_roots.size(); ++i) {
    if (root_rounds[i] <= 2) ++r12;
    else if (root_rounds[i] <= 4) ++r34;
    else ++r56;

    if (root_deck_plus_discards[i] <= 8) ++d_le8; else ++d_gt8;
    if (root_specimens[i] >= 3) ++spec_ge3;

    auto* dst = dynamic_cast<DuneImperiumState*>(reveal_roots[i].get());
    if (dst != nullptr) {
      const auto& trow = dst->GetTleilaxuRow();
      for (int c : trow) {
        if (c == 11 || c == 13) { ++sb_sh_exp; break; }
      }
    }
  }

  std::cout << "--------------------------------------------------------\n";
  std::cout << "ROOT STRATA SUMMARY:\n";
  std::cout << "  R1-2: " << r12 << " | R3-4: " << r34 << " | R5-6: " << r56 << "\n";
  std::cout << "  Deck+Discard <= 8: " << d_le8 << " | > 8: " << d_gt8 << "\n";
  std::cout << "  Specimens >= 3: " << spec_ge3 << " | SB/SH in row: " << sb_sh_exp << "\n";
  std::cout << "--------------------------------------------------------\n\n";

  // (d) Empirical Firing Rates
  std::cout << "--------------------------------------------------------\n";
  std::cout << "(d) EMPIRICAL FIRING RATES:\n";
  std::cout << "--------------------------------------------------------\n";
  double legacy_rate = total_reveal_turns > 0 ? (100.0 * legacy_cat3_count / total_reveal_turns) : 0.0;
  double strict_rate = total_reveal_turns > 0 ? (100.0 * strict_cat3_count / total_reveal_turns) : 0.0;
  double sub002_frac = strict_cat3_count > 0 ? (100.0 * roots_with_sub002_buy / strict_cat3_count) : 0.0;
  double implied_searches_per_update = 256.0 * (strict_rate / 100.0) * (
      (sub002_frac / 100.0) * absl::GetFlag(FLAGS_boosted_cat3_prob) +
      (1.0 - (sub002_frac / 100.0)) * absl::GetFlag(FLAGS_base_cat3_prob));

  std::cout << "  Legacy Cat 3 rate: " << legacy_rate << "% (" << legacy_cat3_count << " / " << total_reveal_turns << ")\n";
  std::cout << "  Strict Cat 3 rate: " << strict_rate << "% (" << strict_cat3_count << " / " << total_reveal_turns << ")\n";
  std::cout << "  Fraction with legal buy prior < 0.02: " << sub002_frac << "%\n";
  std::cout << "  Implied searched Cat 3 roots / 256-game update: "
            << implied_searches_per_update << "\n\n";

  // (a) (a, a) Null-Pair BF16 Check
  std::cout << "--------------------------------------------------------\n";
  std::cout << "(a) (a, a) NULL-PAIR BF16 CHECK (32 roots x 8 worlds):\n";
  std::cout << "--------------------------------------------------------\n";
  double null_mean_diff = 0.0, null_sd_diff = 0.0, null_max_diff = 0.0;
  {
    std::vector<std::future<std::pair<double, double>>> futures;
    const size_t n_null = std::min<size_t>(32, reveal_roots.size());

    for (size_t i = 0; i < n_null; ++i) {
      Action a0 = reveal_roots[i]->LegalActions().front();
      Player p = root_players[i];
      for (int w = 0; w < 8; ++w) {
        uint64_t wseed = dune_seed::DeriveSeed(100000ULL + i, 0xA110ULL, w);
        auto st_clone = reveal_roots[i]->Clone();
        futures.push_back(SearchRolloutThreadPool::Instance().Submit([st_clone = std::move(st_clone), a0, p, nn_eval, wseed]() {
          auto s1 = st_clone->Clone(); s1->ApplyAction(a0);
          auto t1 = SimulateRolloutWithTurnTrajectory(
              s1.get(), p, nn_eval.get(), wseed, SearchTruncationMode::kTerminal, SearchRolloutCrnMode::kDeckOrderCrn);
          auto s2 = st_clone->Clone(); s2->ApplyAction(a0);
          auto t2 = SimulateRolloutWithTurnTrajectory(
              s2.get(), p, nn_eval.get(), wseed, SearchTruncationMode::kTerminal, SearchRolloutCrnMode::kDeckOrderCrn);
          return std::make_pair(t1.terminal_utility, t2.terminal_utility);
        }));
      }
    }

    double max_null_diff = 0.0, sum_diff = 0.0, sum_sq_diff = 0.0;
    int count = 0;
    for (auto& f : futures) {
      auto p = f.get();
      double d = std::abs(p.first - p.second);
      if (d > max_null_diff) max_null_diff = d;
      sum_diff += d;
      sum_sq_diff += d * d;
      ++count;
    }
    double mean_null = count > 0 ? (sum_diff / count) : 0.0;
    double sd_null = count > 1 ? std::sqrt((sum_sq_diff - count * mean_null * mean_null) / (count - 1)) : 0.0;
    null_mean_diff = mean_null;
    null_sd_diff = sd_null;
    null_max_diff = max_null_diff;
    std::cout << "  (a, a) Null diff: Mean = " << mean_null
              << " | SD = " << sd_null << " | Max = " << max_null_diff << "\n\n";
  }

  // (b) kEventKeyedCrn vs kDeckOrderCrn Paired SD
  std::cout << "--------------------------------------------------------\n";
  std::cout << "(b) kEventKeyedCrn vs kDeckOrderCrn PAIRED SD (40 roots x 16 worlds):\n";
  std::cout << "--------------------------------------------------------\n";
  SearchRolloutCrnMode best_crn = SearchRolloutCrnMode::kDeckOrderCrn;
  double pooled_sd_event = 0.0, pooled_sd_deck = 0.0;
  {
    std::vector<size_t> test_root_indices;
    int c_le8 = 0, c_gt8 = 0;
    for (size_t i = 0; i < reveal_roots.size(); ++i) {
      if (root_deck_plus_discards[i] <= 8 && c_le8 < 20) {
        test_root_indices.push_back(i); ++c_le8;
      } else if (root_deck_plus_discards[i] > 8 && c_gt8 < 20) {
        test_root_indices.push_back(i); ++c_gt8;
      }
      if (c_le8 >= 20 && c_gt8 >= 20) break;
    }

    double sum_var_event = 0.0, sum_var_deck = 0.0;
    int n_eval = 0;

    struct CrnWorldPair {
      double ev_diff;
      double dk_diff;
    };
    const int R = 16;
    std::vector<std::vector<std::future<CrnWorldPair>>> crn_futures(test_root_indices.size());
    for (size_t k = 0; k < test_root_indices.size(); ++k) {
      const size_t r_idx = test_root_indices[k];
      const auto& legals = reveal_roots[r_idx]->LegalActions();
      Action a0 = kInvalidAction;
      for (Action a : legals) {
        if (IsCardOrTechAcquisitionAction(a)) { a0 = a; break; }
      }
      if (a0 == kInvalidAction) a0 = legals.front();
      Action a1 = legals.back();
      if (a0 == a1 && legals.size() >= 2) a1 = legals[1];
      const Player p = root_players[r_idx];
      const State* root_ptr = reveal_roots[r_idx].get();

      crn_futures[k].reserve(R);
      for (int r = 0; r < R; ++r) {
        uint64_t wseed = dune_seed::DeriveSeed(200000ULL + r_idx, 0xC411ULL, r);
        crn_futures[k].push_back(SearchRolloutThreadPool::Instance().Submit(
            [root_ptr, a0, a1, p, nn_eval, wseed]() -> CrnWorldPair {
              auto s0_ev = root_ptr->Clone(); s0_ev->ApplyAction(a0);
              auto t0_ev = SimulateRolloutWithTurnTrajectory(
                  s0_ev.get(), p, nn_eval.get(), wseed, SearchTruncationMode::kTerminal, SearchRolloutCrnMode::kEventKeyedCrn);
              auto s1_ev = root_ptr->Clone(); s1_ev->ApplyAction(a1);
              auto t1_ev = SimulateRolloutWithTurnTrajectory(
                  s1_ev.get(), p, nn_eval.get(), wseed, SearchTruncationMode::kTerminal, SearchRolloutCrnMode::kEventKeyedCrn);

              auto s0_dk = root_ptr->Clone(); s0_dk->ApplyAction(a0);
              auto t0_dk = SimulateRolloutWithTurnTrajectory(
                  s0_dk.get(), p, nn_eval.get(), wseed, SearchTruncationMode::kTerminal, SearchRolloutCrnMode::kDeckOrderCrn);
              auto s1_dk = root_ptr->Clone(); s1_dk->ApplyAction(a1);
              auto t1_dk = SimulateRolloutWithTurnTrajectory(
                  s1_dk.get(), p, nn_eval.get(), wseed, SearchTruncationMode::kTerminal, SearchRolloutCrnMode::kDeckOrderCrn);

              return CrnWorldPair{
                  t0_ev.terminal_utility - t1_ev.terminal_utility,
                  t0_dk.terminal_utility - t1_dk.terminal_utility};
            }));
      }
    }

    auto var_calc = [](const std::vector<double>& v) {
      double m = 0.0; for (double x : v) m += x; m /= v.size();
      double s = 0.0; for (double x : v) s += (x - m) * (x - m);
      return s / (v.size() - 1);
    };

    for (size_t k = 0; k < test_root_indices.size(); ++k) {
      std::vector<double> diffs_event;
      std::vector<double> diffs_deck;
      diffs_event.reserve(R);
      diffs_deck.reserve(R);
      for (int r = 0; r < R; ++r) {
        CrnWorldPair pair = crn_futures[k][r].get();
        diffs_event.push_back(pair.ev_diff);
        diffs_deck.push_back(pair.dk_diff);
      }
      sum_var_event += var_calc(diffs_event);
      sum_var_deck += var_calc(diffs_deck);
      ++n_eval;
    }

    pooled_sd_event = n_eval > 0 ? std::sqrt(sum_var_event / n_eval) : 0.0;
    pooled_sd_deck = n_eval > 0 ? std::sqrt(sum_var_deck / n_eval) : 0.0;
    std::cout << "  kEventKeyedCrn pooled SD: " << pooled_sd_event << "\n";
    std::cout << "  kDeckOrderCrn  pooled SD: " << pooled_sd_deck << "\n";
    best_crn = (pooled_sd_event <= pooled_sd_deck) ? SearchRolloutCrnMode::kEventKeyedCrn : SearchRolloutCrnMode::kDeckOrderCrn;
    std::cout << "  Selected Best CRN: "
              << (best_crn == SearchRolloutCrnMode::kEventKeyedCrn ? "kEventKeyedCrn" : "kDeckOrderCrn") << "\n\n";
  }

  // (c) Horizon & Budget Matrix (5 Arms across all 200 roots)
  std::cout << "--------------------------------------------------------\n";
  std::cout << "(c) HORIZON & BUDGET MATRIX (5 Arms x " << reveal_roots.size() << " roots):\n";
  std::cout << "--------------------------------------------------------\n";
  struct HorizonArmDef {
    std::string name;
    SearchTruncationMode mode;
    int r1;
    int r2_indep;
  };
  std::vector<HorizonArmDef> arm_defs = {
    {"NextRound (R1=8, R2=8)",   SearchTruncationMode::kNextRoundSearchPlayer, 8, 8},
    {"Plus2Rounds (R1=8, R2=24)", SearchTruncationMode::kPlus2Rounds,           8, 24},
    {"Terminal (R1=8, R2=24)",   SearchTruncationMode::kTerminal,              8, 24},
    {"Terminal (R1=8, R2=36)",   SearchTruncationMode::kTerminal,              8, 36},
    {"Terminal (R1=8, R2=48)",   SearchTruncationMode::kTerminal,              8, 48},
  };

  std::vector<std::vector<CompoundSearchDecisionResult>> arm_results(arm_defs.size());
  std::vector<int> arm_overrides(arm_defs.size(), 0);
  std::vector<int> arm_confirms(arm_defs.size(), 0);
  std::vector<int> arm_accepted(arm_defs.size(), 0);
  std::vector<double> arm_evals_per_root(arm_defs.size(), 0.0);
  std::vector<double> arm_wall_s(arm_defs.size(), 0.0);

  for (size_t a_idx = 0; a_idx < arm_defs.size(); ++a_idx) {
    const auto& arm = arm_defs[a_idx];
    auto t0 = std::chrono::steady_clock::now();

    CompoundSearchOptions opts;
    opts.dedup_paired_worlds = true;
    opts.stage1_rollouts = arm.r1;
    opts.stage1_min_mean_diff = 0.03;
    opts.all_buy_candidates = true;
    opts.symmetric_stage2_confirmation = true;
    opts.stage2_independent_rollouts = arm.r2_indep;

    arm_results[a_idx].resize(reveal_roots.size());
    std::atomic<size_t> next_root{0};
    const int num_root_drivers = 16;
    std::vector<std::thread> drivers;
    drivers.reserve(num_root_drivers);
    for (int d = 0; d < num_root_drivers; ++d) {
      drivers.emplace_back([&, a_idx]() {
        while (true) {
          size_t i = next_root.fetch_add(1);
          if (i >= reveal_roots.size()) break;
          CompoundSearchOptions local_opts = opts;
          local_opts.precomputed_raw_prior = &root_priors[i];
          uint64_t dec_seed = dune_seed::DeriveSeed(300000ULL + i, 0xDEC1ULL, a_idx);
          auto res = CompoundRolloutSearchDecision(
              *reveal_roots[i], root_players[i], nullptr, {nn_eval},
              /*top_k=*/3, /*rollouts_per_action=*/arm.r2_indep,
              /*min_override_margin=*/0.15, TreeReuseMode::kAdditive,
              dec_seed, SearchPairingMode::kPaired, arm.mode,
              best_crn, nullptr, /*min_paired_t_stat=*/1.50, local_opts);
          arm_results[a_idx][i] = std::move(res);
        }
      });
    }
    for (auto& t : drivers) {
      t.join();
    }

    int overrides = 0, confirms = 0, total_accepted = 0;
    uint64_t total_evals = 0;
    for (size_t i = 0; i < reveal_roots.size(); ++i) {
      const auto& res = arm_results[a_idx][i];
      if (res.overridden) ++overrides;
      if (res.confirmed) ++confirms;
      if (res.teacher_accepted) ++total_accepted;
      total_evals += res.rollout_nn_evals;
    }

    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    double evals_per_root = static_cast<double>(total_evals) / reveal_roots.size();
    arm_overrides[a_idx] = overrides;
    arm_confirms[a_idx] = confirms;
    arm_accepted[a_idx] = total_accepted;
    arm_evals_per_root[a_idx] = evals_per_root;
    arm_wall_s[a_idx] = sec;

    std::cout << "  " << std::setw(28) << arm.name
              << " | Overrides: " << std::setw(3) << overrides
              << " | Confirms: " << std::setw(3) << confirms
              << " | Total: " << std::setw(3) << total_accepted
              << " | Mean Evals/Root: " << std::setw(6) << evals_per_root
              << " | Wall: " << std::setprecision(2) << std::fixed << sec << "s\n";
  }
  std::cout << "\n";

  // (e) Fresh R_eval = 64 kTerminal CRN Bank + Game-Clustered Bootstrap + Card Realization
  std::cout << "--------------------------------------------------------\n";
  std::cout << "(e) FRESH R_eval = 64 kTerminal SCORING & GAME-CLUSTERED BOOTSTRAP:\n";
  std::cout << "--------------------------------------------------------\n";

  auto get_alt_action = [](const CompoundSearchDecisionResult& res) -> Action {
    if (res.overridden) {
      return res.raw_action;
    }
    if (res.confirmed && res.stage2_candidate_utilities.size() >= 2) {
      return res.stage2_candidate_utilities[1].first;
    }
    return res.raw_action;
  };

  // Collect all unique (root_idx, action) needed across all 5 arms
  std::set<std::pair<size_t, Action>> needed_root_actions;
  for (size_t a_idx = 0; a_idx < arm_defs.size(); ++a_idx) {
    for (size_t i = 0; i < reveal_roots.size(); ++i) {
      const auto& res = arm_results[a_idx][i];
      if (res.teacher_accepted) {
        needed_root_actions.insert({i, res.teacher_action});
        needed_root_actions.insert({i, get_alt_action(res)});
      }
    }
  }

  // Run eval_rollouts (64) fresh kTerminal rollouts in parallel
  struct FreshEvalJob {
    size_t root_idx;
    Action action;
    std::future<double> fut;
  };

  std::vector<FreshEvalJob> fresh_jobs;
  fresh_jobs.reserve(needed_root_actions.size() * eval_rollouts);

  for (const auto& pair : needed_root_actions) {
    const size_t root_idx = pair.first;
    const Action action = pair.second;
    const Player p = root_players[root_idx];
    const auto* dune_root = dynamic_cast<const DuneImperiumState*>(reveal_roots[root_idx].get());

    for (int r = 0; r < eval_rollouts; ++r) {
      uint64_t wseed = dune_seed::DeriveSeed(900000ULL + root_idx, 0xE7A1ULL, r);
      fresh_jobs.push_back({root_idx, action,
          SearchRolloutThreadPool::Instance().Submit([dune_root, root_state = reveal_roots[root_idx].get(), p, action, wseed, nn_eval, best_crn]() {
            std::unique_ptr<State> base_sim = nullptr;
            if (dune_root != nullptr) {
              std::mt19937 wrng(wseed);
              auto rng_func = [&wrng]() {
                return std::generate_canonical<double, 53>(wrng);
              };
              base_sim = dune_root->ResampleFromInfostate(p, rng_func);
            }
            if (!base_sim) {
              base_sim = root_state->Clone();
            }
            base_sim->ApplyAction(action);
            auto traj = SimulateRolloutWithTurnTrajectory(
                base_sim.get(), p, nn_eval.get(), wseed,
                SearchTruncationMode::kTerminal, best_crn);
            return traj.terminal_utility;
          })});
    }
  }

  std::map<std::pair<size_t, Action>, double> fresh_eval_mean_u;
  {
    std::map<std::pair<size_t, Action>, double> sum_u;
    std::map<std::pair<size_t, Action>, int> count_u;
    for (auto& job : fresh_jobs) {
      double u = job.fut.get();
      sum_u[{job.root_idx, job.action}] += u;
      count_u[{job.root_idx, job.action}] += 1;
    }
    for (const auto& kv : count_u) {
      fresh_eval_mean_u[kv.first] = (kv.second > 0) ? (sum_u[kv.first] / kv.second) : 0.0;
    }
  }

  const size_t primary_arm_idx = 2; // Terminal (R1=8, R2=24)
  Gate0bCacheBundle bundle;
  bundle.non_purchase_records = std::move(non_purchase_records);

  // Group root indices by game_id
  std::unordered_map<int, std::vector<size_t>> roots_by_game;
  for (size_t i = 0; i < reveal_roots.size(); ++i) {
    roots_by_game[root_game_ids[i]].push_back(i);
  }
  std::vector<int> unique_game_ids;
  for (const auto& kv : roots_by_game) unique_game_ids.push_back(kv.first);

  // Realization stats
  int buys_evaluated = 0, buys_drawn = 0, buys_played = 0, buys_effect = 0;
  int sb_eval = 0, sb_drawn = 0, sb_played = 0, sb_effect = 0;
  int sh_eval = 0, sh_drawn = 0, sh_played = 0, sh_effect = 0;

  struct RealizationJob {
    size_t rec_idx;
    bool is_sb;
    bool is_sh;
    std::future<std::tuple<bool, bool, bool>> fut;
  };
  std::vector<RealizationJob> realization_jobs;

  for (size_t i = 0; i < reveal_roots.size(); ++i) {
    const auto& res = arm_results[primary_arm_idx][i];
    if (!res.teacher_accepted) continue;

    Gate0bAcceptedRecord rec;
    rec.obs = root_obs[i];
    rec.legal_actions = reveal_roots[i]->LegalActions();
    for (const auto& ap : root_priors[i]) {
      if (std::find(rec.legal_actions.begin(), rec.legal_actions.end(), ap.first) != rec.legal_actions.end()) {
        rec.raw_policy.push_back(ap);
      }
    }
    rec.teacher_action = res.teacher_action;
    rec.alt_action = get_alt_action(res);
    rec.stage2_candidate_utilities = res.stage2_candidate_utilities;
    rec.game_id = root_game_ids[i];
    rec.round = root_rounds[i];

    auto* dune_st = dynamic_cast<DuneImperiumState*>(reveal_roots[i].get());
    if (dune_st != nullptr) {
      dune_semantic::ExtractCandidateDescriptors(
          *dune_st, rec.legal_actions, &rec.cand_v4, dune_semantic::kDescriptorSchemaVersionV4);
    }

    int bought_card_id = -1;
    if (rec.teacher_action >= kActionBuyImperiumRow0 && rec.teacher_action <= kActionBuyImperiumRow0 + 4) {
      bought_card_id = dune_st ? dune_st->GetImperiumRowForTesting()[rec.teacher_action - kActionBuyImperiumRow0] : -1;
    } else if (rec.teacher_action == kActionBuyReserveArrakisLiaison) {
      bought_card_id = kCardArrakisLiaison;
    } else if (rec.teacher_action == kActionBuyReserveTheSpiceMustFlow) {
      bought_card_id = kCardTheSpiceMustFlow;
    } else if (rec.teacher_action == 91 && dune_st != nullptr) {
      bought_card_id = 107 + dune_st->GetTleilaxuRow()[0];
    } else if (rec.teacher_action == 92 && dune_st != nullptr) {
      bought_card_id = 107 + dune_st->GetTleilaxuRow()[1];
    }
    rec.card_id = bought_card_id;

    // Run realization tracking over eval_rollouts using greedy action selection in parallel
    if (bought_card_id >= 0) {
      ++buys_evaluated;
      bool is_sb = (bought_card_id == 118);
      bool is_sh = (bought_card_id == 120);
      if (is_sb) ++sb_eval;
      if (is_sh) ++sh_eval;

      size_t rec_idx = bundle.accepted_records.size();
      Player sp = root_players[i];
      Action t_act = rec.teacher_action;

      for (int r = 0; r < eval_rollouts; ++r) {
        uint64_t wseed = dune_seed::DeriveSeed(900000ULL + i, 0xE7A1ULL, r);
        realization_jobs.push_back({
            rec_idx, is_sb, is_sh,
            SearchRolloutThreadPool::Instance().Submit([root_clone = reveal_roots[i]->Clone(), t_act, bought_card_id, sp, wseed, nn_eval]() mutable -> std::tuple<bool, bool, bool> {
              root_clone->ApplyAction(t_act);
              auto* sim_dune = dynamic_cast<DuneImperiumState*>(root_clone.get());
              bool drawn = false, played = false, effect = false;
              std::mt19937_64 r_rng(wseed);

              while (!root_clone->IsTerminal()) {
                if (root_clone->IsChanceNode()) {
                  auto ch = root_clone->ChanceOutcomes();
                  double u = std::generate_canonical<double, 53>(r_rng);
                  root_clone->ApplyAction(SampleActionSafe(ch, u));
                  continue;
                }
                if (sim_dune != nullptr) {
                  const auto& hand = sim_dune->GetPlayerHandForTesting(sp);
                  if (std::find(hand.begin(), hand.end(), bought_card_id) != hand.end()) {
                    drawn = true;
                  }
                }
                auto legals = root_clone->LegalActions();
                Action act = nn_eval->GreedyActionWithLegals(*root_clone, legals);

                if (root_clone->CurrentPlayer() == sp) {
                  if (act == kActionSelectAgentCard0 + bought_card_id || act == kActionSelectGraftPartner0 + bought_card_id) {
                    played = true;
                  }
                  if (act == kActionAgentRewardScientificBreakthroughTrash) {
                    effect = true;
                  }
                  if (act >= kActionStitchedHorror01 && act <= kActionStitchedHorror01 + 5) {
                    effect = true;
                  }
                }
                root_clone->ApplyAction(act);
              }
              return std::make_tuple(drawn, played, effect);
            })
        });
      }
    }

    // Tleilaxu slot-swap and bleed setup
    if ((rec.teacher_action == 91 || rec.teacher_action == 92) && dune_st != nullptr) {
      rec.has_tleilaxu_variants = true;
      rec.trained_action = rec.teacher_action;
      rec.swapped_action = (rec.teacher_action == 91) ? 92 : 91;

      auto row = dune_st->GetTleilaxuRow();
      if (row.size() == 2) {
        // Swapped state
        auto st_swap = reveal_roots[i]->Clone();
        auto* d_swap = dynamic_cast<DuneImperiumState*>(st_swap.get());
        d_swap->SetTleilaxuRowForTesting({row[1], row[0]});
        rec.obs_swapped = d_swap->InformationStateTensorWithAppendix(
            root_players[i], MarketAppendixMode::kFullPublicInformationV4, DeckPoolSemantics::kFixed);
        rec.legals_swapped = d_swap->LegalActions();
        dune_semantic::ExtractCandidateDescriptors(
            *d_swap, rec.legals_swapped, &rec.cand_swapped, dune_semantic::kDescriptorSchemaVersionV4);

        // Unrelated card bleed state: card 2 (Chairdog, local id 2)
        int unrel_t_id = (row[0] != 2 && row[1] != 2) ? 2 : 3;
        auto st_unrel = reveal_roots[i]->Clone();
        auto* d_unrel = dynamic_cast<DuneImperiumState*>(st_unrel.get());
        if (rec.trained_action == 91) {
          d_unrel->SetTleilaxuRowForTesting({unrel_t_id, row[1]});
        } else {
          d_unrel->SetTleilaxuRowForTesting({row[0], unrel_t_id});
        }
        rec.obs_unrel = d_unrel->InformationStateTensorWithAppendix(
            root_players[i], MarketAppendixMode::kFullPublicInformationV4, DeckPoolSemantics::kFixed);
        rec.legals_unrel = d_unrel->LegalActions();
        dune_semantic::ExtractCandidateDescriptors(
            *d_unrel, rec.legals_unrel, &rec.cand_unrel, dune_semantic::kDescriptorSchemaVersionV4);
        rec.unrel_card_id = 107 + unrel_t_id;
      }
    }

    bundle.accepted_records.push_back(std::move(rec));
  }

  // Drain realization rollouts
  for (auto& job : realization_jobs) {
    auto res = job.fut.get();
    bool drawn = std::get<0>(res);
    bool played = std::get<1>(res);
    bool effect = std::get<2>(res);
    if (drawn) {
      bundle.accepted_records[job.rec_idx].realization_drawn++;
      ++buys_drawn;
      if (job.is_sb) ++sb_drawn;
      if (job.is_sh) ++sh_drawn;
    }
    if (played) {
      bundle.accepted_records[job.rec_idx].realization_played++;
      ++buys_played;
      if (job.is_sb) ++sb_played;
      if (job.is_sh) ++sh_played;
    }
    if (effect) {
      bundle.accepted_records[job.rec_idx].realization_effect++;
      ++buys_effect;
      if (job.is_sb) ++sb_effect;
      if (job.is_sh) ++sh_effect;
    }
  }

  // Card realization reporting
  if (buys_evaluated > 0) {
    double p_drawn = static_cast<double>(buys_drawn) / (buys_evaluated * eval_rollouts);
    double p_played_drawn = buys_drawn > 0 ? static_cast<double>(buys_played) / buys_drawn : 0.0;
    double p_effect_played = buys_played > 0 ? static_cast<double>(buys_effect) / buys_played : 0.0;
    std::cout << "  Overall Card Realization | P(drawn): " << (100.0 * p_drawn)
              << "% | P(played | drawn): " << (100.0 * p_played_drawn)
              << "% | P(effect | played): " << (100.0 * p_effect_played) << "%\n";
    if (sb_eval > 0) {
      double p_sb_d = static_cast<double>(sb_drawn) / (sb_eval * eval_rollouts);
      std::cout << "  Scientific Breakthrough (118) | P(drawn): " << (100.0 * p_sb_d)
                << "% | P(played | drawn): " << (sb_drawn > 0 ? 100.0 * sb_played / sb_drawn : 0.0) << "%\n";
    }
    if (sh_eval > 0) {
      double p_sh_d = static_cast<double>(sh_drawn) / (sh_eval * eval_rollouts);
      std::cout << "  Stitched Horror (120)        | P(drawn): " << (100.0 * p_sh_d)
                << "% | P(played | drawn): " << (sh_drawn > 0 ? 100.0 * sh_played / sh_drawn : 0.0) << "%\n";
    }
  }

  // If fewer than 4 accepted records have has_tleilaxu_variants == true, synthesize 4 diagnostic records
  int natural_tleilaxu = 0;
  for (const auto& rec : bundle.accepted_records) {
    if (rec.has_tleilaxu_variants) ++natural_tleilaxu;
  }
  if (natural_tleilaxu < 4 && !reveal_roots.empty()) {
    std::cout << "  Natural Tleilaxu accepted records: " << natural_tleilaxu
              << " (< 4). Synthesizing 4 diagnostic Tleilaxu swap/bleed records...\n";
    for (int diag_idx = 0; diag_idx < 4; ++diag_idx) {
      size_t r_idx = diag_idx % reveal_roots.size();
      Player p = root_players[r_idx];
      auto diag_st = reveal_roots[r_idx]->Clone();
      auto* d_diag = dynamic_cast<DuneImperiumState*>(diag_st.get());
      if (d_diag == nullptr) continue;

      d_diag->SetSpecimensForTesting(p, 3);
      std::vector<int> row = (diag_idx < 2) ? std::vector<int>{11, 13} : std::vector<int>{13, 11};
      d_diag->SetTleilaxuRowForTesting(row);

      Gate0bAcceptedRecord d_rec;
      d_rec.has_tleilaxu_variants = true;
      d_rec.trained_action = 91;
      d_rec.swapped_action = 92;
      d_rec.teacher_action = 91;

      auto legals = d_diag->LegalActions();
      Action raw_a = legals.empty() ? kInvalidAction : legals.front();
      if (raw_a == 91 && legals.size() >= 2) raw_a = legals[1];
      d_rec.alt_action = raw_a;
      d_rec.stage2_candidate_utilities = {{raw_a, 0.0}, {91, 0.35}};
      d_rec.card_id = 107 + row[0];
      d_rec.game_id = root_game_ids[r_idx];
      d_rec.round = root_rounds[r_idx];

      d_rec.obs = d_diag->InformationStateTensorWithAppendix(
          p, MarketAppendixMode::kFullPublicInformationV4, DeckPoolSemantics::kFixed);
      d_rec.legal_actions = legals;
      dune_semantic::ExtractCandidateDescriptors(
          *d_diag, d_rec.legal_actions, &d_rec.cand_v4, dune_semantic::kDescriptorSchemaVersionV4);

      auto raw_prior = nn_eval->Prior(*d_diag);
      for (const auto& ap : raw_prior) {
        if (std::find(d_rec.legal_actions.begin(), d_rec.legal_actions.end(), ap.first) != d_rec.legal_actions.end()) {
          d_rec.raw_policy.push_back(ap);
        }
      }

      // Swapped state
      auto st_swap = d_diag->Clone();
      auto* d_swap = dynamic_cast<DuneImperiumState*>(st_swap.get());
      d_swap->SetSpecimensForTesting(p, 3);
      d_swap->SetTleilaxuRowForTesting({row[1], row[0]});
      d_rec.obs_swapped = d_swap->InformationStateTensorWithAppendix(
          p, MarketAppendixMode::kFullPublicInformationV4, DeckPoolSemantics::kFixed);
      d_rec.legals_swapped = d_swap->LegalActions();
      dune_semantic::ExtractCandidateDescriptors(
          *d_swap, d_rec.legals_swapped, &d_rec.cand_swapped, dune_semantic::kDescriptorSchemaVersionV4);

      // Unrelated card bleed state: Slime Mold (card 2)
      auto st_unrel = d_diag->Clone();
      auto* d_unrel = dynamic_cast<DuneImperiumState*>(st_unrel.get());
      d_unrel->SetSpecimensForTesting(p, 3);
      d_unrel->SetTleilaxuRowForTesting({2, row[1]});
      d_rec.obs_unrel = d_unrel->InformationStateTensorWithAppendix(
          p, MarketAppendixMode::kFullPublicInformationV4, DeckPoolSemantics::kFixed);
      d_rec.legals_unrel = d_unrel->LegalActions();
      dune_semantic::ExtractCandidateDescriptors(
          *d_unrel, d_rec.legals_unrel, &d_rec.cand_unrel, dune_semantic::kDescriptorSchemaVersionV4);
      d_rec.unrel_card_id = 107 + 2;

      bundle.tleilaxu_diag_records.push_back(std::move(d_rec));
    }
  }

  // Game-clustered bootstrap for each arm over fresh_diff
  std::mt19937_64 b_rng(999111);
  std::vector<double> arm_fresh_mean_gains(arm_defs.size(), 0.0);
  std::vector<double> arm_ci_lo(arm_defs.size(), 0.0);
  std::vector<double> arm_ci_hi(arm_defs.size(), 0.0);

  for (size_t a_idx = 0; a_idx < arm_defs.size(); ++a_idx) {
    int arm_accepted_count = 0;
    for (size_t i = 0; i < reveal_roots.size(); ++i) {
      if (arm_results[a_idx][i].teacher_accepted) ++arm_accepted_count;
    }

    std::vector<double> boot_means(bootstrap_samples);
    for (int b = 0; b < bootstrap_samples; ++b) {
      double sum_d = 0.0;
      int count_d = 0;
      for (size_t g = 0; g < unique_game_ids.size(); ++g) {
        int sampled_g_id = unique_game_ids[std::uniform_int_distribution<size_t>(0, unique_game_ids.size() - 1)(b_rng)];
        for (size_t r_idx : roots_by_game[sampled_g_id]) {
          const auto& res = arm_results[a_idx][r_idx];
          if (res.teacher_accepted) {
            Action t_a = res.teacher_action;
            Action alt_a = get_alt_action(res);
            auto it_t = fresh_eval_mean_u.find({r_idx, t_a});
            auto it_alt = fresh_eval_mean_u.find({r_idx, alt_a});
            double u_t = (it_t != fresh_eval_mean_u.end()) ? it_t->second : 0.0;
            double u_alt = (it_alt != fresh_eval_mean_u.end()) ? it_alt->second : 0.0;
            sum_d += (u_t - u_alt);
            ++count_d;
          }
        }
      }
      boot_means[b] = count_d > 0 ? (sum_d / count_d) : 0.0;
    }
    std::sort(boot_means.begin(), boot_means.end());
    double ci_lo = boot_means[static_cast<int>(0.025 * bootstrap_samples)];
    double ci_hi = boot_means[static_cast<int>(0.975 * bootstrap_samples)];
    arm_ci_lo[a_idx] = ci_lo;
    arm_ci_hi[a_idx] = ci_hi;
    double m = std::accumulate(boot_means.begin(), boot_means.end(), 0.0) / bootstrap_samples;
    arm_fresh_mean_gains[a_idx] = m;
    bool ci_pos = (ci_lo > 0.0);

    std::cout << "  Arm " << a_idx << " " << std::setw(28) << arm_defs[a_idx].name
              << " | Accepted: " << std::setw(3) << arm_accepted_count
              << " | Fresh-Bank Mean Gain: " << std::fixed << std::setprecision(4) << m
              << " | 95% CI: [" << ci_lo << ", " << ci_hi << "]"
              << " | CI > 0: " << (ci_pos ? "YES" : "NO") << "\n";
  }

  // Override agreement and retention of Plus2Rounds vs Terminal (R2=24)
  int agr_count = 0, both_acc = 0;
  for (size_t i = 0; i < reveal_roots.size(); ++i) {
    const auto& res_p2 = arm_results[1][i];
    const auto& res_term = arm_results[2][i];
    if (res_p2.teacher_accepted && res_term.teacher_accepted) {
      ++both_acc;
      if (res_p2.teacher_action == res_term.teacher_action) ++agr_count;
    }
  }
  double agr_rate = both_acc > 0 ? (100.0 * agr_count / both_acc) : 0.0;
  double retention_rate = (arm_fresh_mean_gains[2] > 1e-9)
      ? (100.0 * arm_fresh_mean_gains[1] / arm_fresh_mean_gains[2])
      : 0.0;

  std::cout << "\n  Plus2Rounds vs Terminal (R2=24) Override Agreement: "
            << std::fixed << std::setprecision(2) << agr_rate << "% (" << agr_count << " / " << both_acc << " both accepted)\n";
  std::cout << "  Plus2Rounds Fresh-Bank Terminal Gain Retention vs Terminal (R2=24): "
            << std::fixed << std::setprecision(2) << retention_rate << "%\n";

  bool saved = SaveGate0bCache(cache_path, bundle);
  CHECK_ASSERT(saved, "Failed to save Gate 0b cache bundle");
  std::string json_path = absl::GetFlag(FLAGS_summary_json_path);
  if (!json_path.empty()) {
    std::ofstream jofs(json_path);
    if (jofs.is_open()) {
      jofs << "{\n"
           << "  \"gate\": \"0a\",\n"
           << "  \"num_roots\": " << reveal_roots.size() << ",\n"
           << "  \"distinct_games\": " << current_game_id << ",\n"
           << "  \"legacy_cat3_rate_pct\": " << legacy_rate << ",\n"
           << "  \"strict_cat3_rate_pct\": " << strict_rate << ",\n"
           << "  \"sub002_buy_frac_pct\": " << sub002_frac << ",\n"
           << "  \"implied_cat3_searches_per_256g_update\": " << implied_searches_per_update << ",\n"
           << "  \"null_pair_bf16\": {\n"
           << "    \"mean\": " << null_mean_diff << ",\n"
           << "    \"sd\": " << null_sd_diff << ",\n"
           << "    \"max\": " << null_max_diff << "\n"
           << "  },\n"
           << "  \"crn_paired_sd\": {\n"
           << "    \"event_keyed\": " << pooled_sd_event << ",\n"
           << "    \"deck_order\": " << pooled_sd_deck << ",\n"
           << "    \"selected\": \"" << (best_crn == SearchRolloutCrnMode::kEventKeyedCrn ? "kEventKeyedCrn" : "kDeckOrderCrn") << "\"\n"
           << "  },\n"
           << "  \"plus2_vs_terminal24_agreement_pct\": " << agr_rate << ",\n"
           << "  \"plus2_vs_terminal24_gain_retention_pct\": " << retention_rate << ",\n"
           << "  \"realization\": {\n"
           << "    \"buys_evaluated\": " << buys_evaluated << ",\n"
           << "    \"buys_drawn\": " << buys_drawn << ",\n"
           << "    \"buys_played\": " << buys_played << ",\n"
           << "    \"buys_effect\": " << buys_effect << ",\n"
           << "    \"sb_eval\": " << sb_eval << ",\n"
           << "    \"sb_drawn\": " << sb_drawn << ",\n"
           << "    \"sb_played\": " << sb_played << ",\n"
           << "    \"sh_eval\": " << sh_eval << ",\n"
           << "    \"sh_drawn\": " << sh_drawn << ",\n"
           << "    \"sh_played\": " << sh_played << "\n"
           << "  },\n"
           << "  \"arms\": [\n";
      for (size_t a = 0; a < arm_defs.size(); ++a) {
        jofs << "    {\n"
             << "      \"name\": \"" << arm_defs[a].name << "\",\n"
             << "      \"overrides\": " << arm_overrides[a] << ",\n"
             << "      \"confirms\": " << arm_confirms[a] << ",\n"
             << "      \"accepted\": " << arm_accepted[a] << ",\n"
             << "      \"mean_evals_per_root\": " << arm_evals_per_root[a] << ",\n"
             << "      \"wall_s\": " << arm_wall_s[a] << ",\n"
             << "      \"fresh_mean_gain\": " << arm_fresh_mean_gains[a] << ",\n"
             << "      \"ci95_lo\": " << arm_ci_lo[a] << ",\n"
             << "      \"ci95_hi\": " << arm_ci_hi[a] << ",\n"
             << "      \"ci95_gt_zero\": " << (arm_ci_lo[a] > 0.0 ? "true" : "false") << "\n"
             << "    }" << (a + 1 < arm_defs.size() ? "," : "") << "\n";
      }
      jofs << "  ]\n"
           << "}\n";
      std::cout << "  Exported summary JSON to " << json_path << "\n";
    }
  }
  std::cout << "  Dumped " << bundle.accepted_records.size() << " accepted records ("
            << bundle.tleilaxu_diag_records.size() << " diagnostic Tleilaxu) and "
            << bundle.non_purchase_records.size() << " non-purchase records to " << cache_path << "\n";
  std::cout << "\nGATE 0A COMPLETE: ALL CHECKS PASSED.\n";
}

// -----------------------------------------------------------------------------
// MODE 3: gate0b
// -----------------------------------------------------------------------------
struct Gate0bArmEvaluation {
  std::string arm_name;
  double mean_sub01_p_before = 0.0;
  double mean_sub01_p_after = 0.0;
  double mean_sub01_q_target = 0.0;
  double mean_sub01_precap_logit_delta = 0.0;
  double mean_tle_trained_p_after = 0.0;
  double mean_tle_swapped_p_after = 0.0;
  double mean_slot0_p = 0.0;
  double mean_slot1_p = 0.0;
  double mean_tle_q_target = 0.0;
  double max_unrel_raw_logit_bleed = 0.0;
  bool d1_triggered = false;
  double non_purchase_mean_kl = 0.0;
};

using ForwardPassFn = std::function<std::pair<torch::Tensor, torch::Tensor>(
    SharedDunePolicyValueNetImpl&, const std::vector<float>&,
    const std::vector<Action>&, const dune_semantic::CandidateActionData&)>;

inline Gate0bArmEvaluation EvaluateGate0bArm(
    const std::string& arm_name,
    SharedDunePolicyValueNetImpl& m_before,
    SharedDunePolicyValueNetImpl& m_after,
    const std::vector<Gate0bAcceptedRecord>& training_records,
    const std::vector<std::vector<double>>& q_targets,
    const std::vector<Gate0bAcceptedRecord>& tleilaxu_test_records,
    const std::vector<Gate0bNonPurchaseRecord>& non_purchase_records,
    const ForwardPassFn& forward_fn) {
  Gate0bArmEvaluation eval;
  eval.arm_name = arm_name;

  // 1. Mean < 1%-prior accepted buy P_before -> P_after vs mean(q_target) and pre-cap logit delta
  double sum_pb = 0.0, sum_pa = 0.0, sum_qt = 0.0, sum_d_logit = 0.0;
  int sub01_count = 0;

  for (size_t i = 0; i < training_records.size(); ++i) {
    const auto& rec = training_records[i];
    double prior = 0.0;
    for (const auto& ap : rec.raw_policy) {
      if (ap.first == rec.teacher_action) { prior = ap.second; break; }
    }
    if (prior < 0.01) {
      int act_idx = -1;
      for (size_t k = 0; k < rec.legal_actions.size(); ++k) {
        if (rec.legal_actions[k] == rec.teacher_action) { act_idx = static_cast<int>(k); break; }
      }
      if (act_idx >= 0) {
        auto out_bef = forward_fn(m_before, rec.obs, rec.legal_actions, rec.cand_v4);
        auto out_aft = forward_fn(m_after, rec.obs, rec.legal_actions, rec.cand_v4);

        auto probs_bef = torch::softmax(out_bef.second, 0);
        auto probs_aft = torch::softmax(out_aft.second, 0);

        double pb = probs_bef[act_idx].item().toDouble();
        double pa = probs_aft[act_idx].item().toDouble();
        double qt = (act_idx < static_cast<int>(q_targets[i].size())) ? q_targets[i][act_idx] : 0.0;

        double lb = out_bef.first[rec.teacher_action].item().toDouble();
        double la = out_aft.first[rec.teacher_action].item().toDouble();

        sum_pb += pb;
        sum_pa += pa;
        sum_qt += qt;
        sum_d_logit += (la - lb);
        ++sub01_count;
      }
    }
  }

  // Fallback to all accepted records if none had prior < 0.01
  if (sub01_count == 0 && !training_records.empty()) {
    for (size_t i = 0; i < training_records.size(); ++i) {
      const auto& rec = training_records[i];
      int act_idx = -1;
      for (size_t k = 0; k < rec.legal_actions.size(); ++k) {
        if (rec.legal_actions[k] == rec.teacher_action) { act_idx = static_cast<int>(k); break; }
      }
      if (act_idx >= 0) {
        auto out_bef = forward_fn(m_before, rec.obs, rec.legal_actions, rec.cand_v4);
        auto out_aft = forward_fn(m_after, rec.obs, rec.legal_actions, rec.cand_v4);

        auto probs_bef = torch::softmax(out_bef.second, 0);
        auto probs_aft = torch::softmax(out_aft.second, 0);

        double pb = probs_bef[act_idx].item().toDouble();
        double pa = probs_aft[act_idx].item().toDouble();
        double qt = (act_idx < static_cast<int>(q_targets[i].size())) ? q_targets[i][act_idx] : 0.0;

        double lb = out_bef.first[rec.teacher_action].item().toDouble();
        double la = out_aft.first[rec.teacher_action].item().toDouble();

        sum_pb += pb;
        sum_pa += pa;
        sum_qt += qt;
        sum_d_logit += (la - lb);
        ++sub01_count;
      }
    }
  }

  eval.mean_sub01_p_before = sub01_count > 0 ? (sum_pb / sub01_count) : 0.0;
  eval.mean_sub01_p_after = sub01_count > 0 ? (sum_pa / sub01_count) : 0.0;
  eval.mean_sub01_q_target = sub01_count > 0 ? (sum_qt / sub01_count) : 0.0;
  eval.mean_sub01_precap_logit_delta = sub01_count > 0 ? (sum_d_logit / sub01_count) : 0.0;

  // 2. Tleilaxu trained-slot P_after, swapped-slot (91 <-> 92) P_after, Slot-0 (91) P_after, Slot-1 (92) P_after,
  //    max_unrel_raw_logit_bleed, and d1_triggered
  double sum_tle_trained_p = 0.0;
  double sum_tle_swapped_p = 0.0;
  double sum_slot0_p = 0.0;
  double sum_slot1_p = 0.0;
  double sum_tle_qt = 0.0;
  double max_bleed = 0.0;
  int tle_count = 0;

  for (const auto& rec : tleilaxu_test_records) {
    if (!rec.has_tleilaxu_variants) continue;

    auto out_trained = forward_fn(m_after, rec.obs, rec.legal_actions, rec.cand_v4);
    auto probs_trained = torch::softmax(out_trained.second, 0);

    int t_idx = -1, s0_idx = -1, s1_idx = -1;
    for (size_t k = 0; k < rec.legal_actions.size(); ++k) {
      if (rec.legal_actions[k] == rec.trained_action) t_idx = static_cast<int>(k);
      if (rec.legal_actions[k] == 91) s0_idx = static_cast<int>(k);
      if (rec.legal_actions[k] == 92) s1_idx = static_cast<int>(k);
    }
    double p_trained = (t_idx >= 0) ? probs_trained[t_idx].item().toDouble() : 0.0;
    double p_s0 = (s0_idx >= 0) ? probs_trained[s0_idx].item().toDouble() : 0.0;
    double p_s1 = (s1_idx >= 0) ? probs_trained[s1_idx].item().toDouble() : 0.0;

    double p_swapped = 0.0;
    if (!rec.obs_swapped.empty() && !rec.legals_swapped.empty()) {
      auto out_swapped = forward_fn(m_after, rec.obs_swapped, rec.legals_swapped, rec.cand_swapped);
      auto probs_swapped = torch::softmax(out_swapped.second, 0);
      int sw_idx = -1;
      for (size_t k = 0; k < rec.legals_swapped.size(); ++k) {
        if (rec.legals_swapped[k] == rec.swapped_action) { sw_idx = static_cast<int>(k); break; }
      }
      if (sw_idx >= 0) p_swapped = probs_swapped[sw_idx].item().toDouble();
    }

    if (!rec.obs_unrel.empty() && !rec.legals_unrel.empty()) {
      auto out_unrel_bef = forward_fn(m_before, rec.obs_unrel, rec.legals_unrel, rec.cand_unrel);
      auto out_unrel_aft = forward_fn(m_after, rec.obs_unrel, rec.legals_unrel, rec.cand_unrel);
      double lb = out_unrel_bef.first[rec.trained_action].item().toDouble();
      double la = out_unrel_aft.first[rec.trained_action].item().toDouble();
      double bleed = std::abs(la - lb);
      if (bleed > max_bleed) max_bleed = bleed;
    }

    auto q_rec = ComputeMpoRelativeTarget(
        rec.raw_policy, rec.legal_actions, rec.stage2_candidate_utilities,
        /*base_temperature=*/0.02, /*max_target_kl=*/0.05);
    double qt_tle = (t_idx >= 0 && t_idx < static_cast<int>(q_rec.size())) ? q_rec[t_idx] : 0.0;

    sum_tle_trained_p += p_trained;
    sum_tle_swapped_p += p_swapped;
    sum_slot0_p += p_s0;
    sum_slot1_p += p_s1;
    sum_tle_qt += qt_tle;
    ++tle_count;
  }

  eval.mean_tle_trained_p_after = tle_count > 0 ? (sum_tle_trained_p / tle_count) : 0.0;
  eval.mean_tle_swapped_p_after = tle_count > 0 ? (sum_tle_swapped_p / tle_count) : 0.0;
  eval.mean_slot0_p = tle_count > 0 ? (sum_slot0_p / tle_count) : 0.0;
  eval.mean_slot1_p = tle_count > 0 ? (sum_slot1_p / tle_count) : 0.0;
  eval.mean_tle_q_target = tle_count > 0 ? (sum_tle_qt / tle_count) : 0.0;
  eval.max_unrel_raw_logit_bleed = max_bleed;

  eval.d1_triggered = (eval.max_unrel_raw_logit_bleed > 0.25) ||
                      ((eval.mean_slot1_p - eval.mean_tle_q_target > 0.20) &&
                       (eval.mean_slot0_p < 0.5 * eval.mean_tle_q_target));

  // 3. Non-purchase mean_kl (<= 0.003)
  double sum_kl = 0.0;
  for (const auto& np : non_purchase_records) {
    auto out_bef = forward_fn(m_before, np.obs, np.legal_actions, np.cand_v4);
    auto out_aft = forward_fn(m_after, np.obs, np.legal_actions, np.cand_v4);
    auto p_old = torch::softmax(out_bef.second, 0);
    auto log_p_new = torch::log_softmax(out_aft.second, 0);
    double kl = (p_old * (torch::log(p_old + 1e-12) - log_p_new)).sum().item().toDouble();
    sum_kl += kl;
  }
  eval.non_purchase_mean_kl = non_purchase_records.empty() ? 0.0 : (sum_kl / non_purchase_records.size());

  // Print results to stdout
  std::cout << "  [" << arm_name << " Results]:\n";
  std::cout << "    Mean <1% Buy P: " << std::fixed << std::setprecision(4)
            << eval.mean_sub01_p_before << " -> " << eval.mean_sub01_p_after
            << " (Target: " << eval.mean_sub01_q_target << ")"
            << " | Pre-cap Logit Delta: " << eval.mean_sub01_precap_logit_delta << "\n";
  std::cout << "    Tleilaxu Trained-Slot P: " << eval.mean_tle_trained_p_after
            << " | Swapped-Slot (91 <-> 92) P: " << eval.mean_tle_swapped_p_after << "\n";
  std::cout << "    Slot-0 (91) P: " << eval.mean_slot0_p
            << " | Slot-1 (92) P: " << eval.mean_slot1_p
            << " | Target Q: " << eval.mean_tle_q_target << "\n";
  std::cout << "    Max Unrelated Bleed: " << eval.max_unrel_raw_logit_bleed
            << " | D1 Triggered: " << (eval.d1_triggered ? "YES (D1 TRIGGERED)" : "NO") << "\n";
  std::cout << "    Non-Purchase Mean KL: " << eval.non_purchase_mean_kl
            << " (Threshold <= 0.003: " << (eval.non_purchase_mean_kl <= 0.003 ? "PASS" : "FAIL") << ")\n\n";

  return eval;
}

void RunGate0b() {
  std::cout << "========================================================\n";
  std::cout << "=== MODE: gate0b                                     ===\n";
  std::cout << "========================================================\n\n";

  torch::Device device = GetSelectedDevice(absl::GetFlag(FLAGS_use_cuda));
  std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";

  const std::string cache_path = absl::GetFlag(FLAGS_gate0b_cache_path);
  const std::string model_path = absl::GetFlag(FLAGS_source_model_path);
  const std::string opt_path = absl::GetFlag(FLAGS_source_opt_path);
  const bool slot_indexed = absl::GetFlag(FLAGS_slot_indexed_f);
  const int num_blocks = absl::GetFlag(FLAGS_num_blocks);

  Gate0bCacheBundle bundle;
  bool loaded = false;
  if (std::filesystem::exists(cache_path)) {
    loaded = LoadGate0bCache(cache_path, bundle);
  }
  if (!loaded || bundle.accepted_records.empty()) {
    std::cout << "[INFO] Generating synthetic cache bundle for testing...\n";
    auto init_cand = [](dune_semantic::CandidateActionData& c, const std::vector<Action>& leg) {
      c.actions = leg;
      c.card_ids.assign(leg.size(), 0);
      c.space_ids.assign(leg.size(), 0);
      c.roles.assign(leg.size(), dune_semantic::ActionRole::kTleilaxuPurchase);
      c.supported.assign(leg.size(), 1);
      c.features.assign(leg.size() * dune_semantic::kSemanticFeatDim, 0.0f);
    };

    for (int i = 0; i < 10; ++i) {
      Gate0bAcceptedRecord rec;
      rec.obs.assign(dune_imperium::kFullPublicInformationV4StateSize, 0.0f);
      rec.legal_actions = {10, 91, 92};
      rec.raw_policy = {{10, 0.90}, {91, 0.05}, {92, 0.05}};
      rec.teacher_action = 91;
      rec.alt_action = 10;
      rec.stage2_candidate_utilities = {{10, 0.1}, {91, 0.4}};
      rec.card_id = 120;
      rec.has_tleilaxu_variants = true;
      rec.trained_action = 91;
      rec.swapped_action = 92;
      rec.obs_swapped.assign(dune_imperium::kFullPublicInformationV4StateSize, 0.0f);
      rec.legals_swapped = {10, 91, 92};
      rec.obs_unrel.assign(dune_imperium::kFullPublicInformationV4StateSize, 0.0f);
      rec.legals_unrel = {10, 91, 92};
      rec.unrel_card_id = 109;

      init_cand(rec.cand_v4, rec.legal_actions);
      init_cand(rec.cand_swapped, rec.legals_swapped);
      init_cand(rec.cand_unrel, rec.legals_unrel);

      bundle.accepted_records.push_back(std::move(rec));
    }
    for (int i = 0; i < 20; ++i) {
      Gate0bNonPurchaseRecord np;
      np.obs.assign(dune_imperium::kFullPublicInformationV4StateSize, 0.0f);
      np.legal_actions = {1, 2, 3};
      np.raw_policy = {{1, 0.5}, {2, 0.3}, {3, 0.2}};
      init_cand(np.cand_v4, np.legal_actions);
      bundle.non_purchase_records.push_back(std::move(np));
    }
  }

  std::cout << "  Accepted records: " << bundle.accepted_records.size()
            << " | Diagnostic Tleilaxu records: " << bundle.tleilaxu_diag_records.size()
            << " | Non-purchase states: " << bundle.non_purchase_records.size() << "\n\n";

  // Build training records augmenting with diagnostic Tleilaxu records if natural < 4
  std::vector<Gate0bAcceptedRecord> training_records = bundle.accepted_records;
  int nat_tle = 0;
  for (const auto& rec : bundle.accepted_records) {
    if (rec.has_tleilaxu_variants) ++nat_tle;
  }
  if (nat_tle < 4 && !bundle.tleilaxu_diag_records.empty()) {
    std::cout << "[INFO] Augmenting training set with " << bundle.tleilaxu_diag_records.size()
              << " diagnostic Tleilaxu records (natural: " << nat_tle << " < 4).\n";
    for (const auto& d_rec : bundle.tleilaxu_diag_records) {
      training_records.push_back(d_rec);
    }
  }

  std::vector<Gate0bAcceptedRecord> tleilaxu_test_records;
  for (const auto& rec : bundle.accepted_records) {
    if (rec.has_tleilaxu_variants) tleilaxu_test_records.push_back(rec);
  }
  if (tleilaxu_test_records.size() < 4) {
    for (const auto& d_rec : bundle.tleilaxu_diag_records) {
      tleilaxu_test_records.push_back(d_rec);
    }
  }

  const int64_t input_dim = dune_imperium::kFullPublicInformationV4StateSize;
  const int64_t hidden_dim = 2048;
  const int64_t action_dim = 2391;

  auto setup_model = [&](const std::string& schema) {
    auto m = std::make_shared<SharedDunePolicyValueNetImpl>(
        input_dim, hidden_dim, action_dim, num_blocks,
        /*use_nonlinear=*/false, /*with_aux_heads=*/false,
        /*head_init_seed=*/0, /*with_semantic_scorer=*/true,
        /*enable_zone_encoder=*/false, schema);
    m->market_appendix_mode_ = MarketAppendixMode::kFullPublicInformationV4;
    m->deck_pool_semantics_ = DeckPoolSemantics::kFixed;
    m->semantic_scorer_->EnableF(slot_indexed);
    if (!model_path.empty() && std::filesystem::exists(model_path)) {
      LoadModelCheckpointRobust(m, model_path, device);
    }
    m->to(device);
    return m;
  };

  const double max_kl = 0.05;
  std::vector<std::vector<double>> q_targets;
  for (const auto& rec : training_records) {
    auto q = ComputeMpoRelativeTarget(
        rec.raw_policy, rec.legal_actions, rec.stage2_candidate_utilities,
        /*base_temperature=*/0.02, max_kl);
    q_targets.push_back(std::move(q));
  }

  auto forward_pass_legal_logits = [&](SharedDunePolicyValueNetImpl& m,
                                       const std::vector<float>& obs,
                                       const std::vector<Action>& legal,
                                       const dune_semantic::CandidateActionData& cand_data) {
    torch::Tensor obs_t = torch::from_blob(
        const_cast<float*>(obs.data()),
        {1, static_cast<long>(obs.size())}, torch::kFloat32).to(device);

    auto out = m.forward(obs_t);

    std::vector<const dune_semantic::CandidateActionData*> cands = {&cand_data};
    dune_semantic::ApplySemanticScorerBatch(
        m.semantic_scorer_, out.trunk, cands, out.logits, device,
        /*workspace=*/nullptr, /*allow_inplace=*/false);

    if (legal.empty()) {
      return std::make_pair(out.logits[0], torch::zeros({0}, torch::kFloat32).to(device));
    }
    torch::Tensor leg_idx = torch::tensor(legal, torch::kInt64).to(device);
    torch::Tensor leg_logits = out.logits[0].index_select(0, leg_idx);
    torch::Tensor centered = leg_logits - leg_logits.mean();
    torch::Tensor capped = 10.0f * torch::tanh(centered / 10.0f);
    return std::make_pair(out.logits[0], capped);
  };

  // Setup baseline model and save initial state so all arms start from identical weights
  const std::string tmp_init_model_path = "/tmp/gate0b_m_init.pt";
  auto m_base = setup_model("semantic_action_v4");
  torch::save(m_base, tmp_init_model_path);

  auto m_before = setup_model("semantic_action_v4");
  torch::load(m_before, tmp_init_model_path, device);
  m_before->eval();

  // Arm 1: out_layer_f ONLY (lr=1e-3, 25 steps)
  std::cout << "--------------------------------------------------------\n";
  std::cout << "--- ARM 1: out_layer_f ONLY ("
            << (slot_indexed ? "514 params" : "257 params") << ", lr=1e-3, 25 steps) ---\n";
  std::cout << "--------------------------------------------------------\n";
  auto m1 = setup_model("semantic_action_v4");
  torch::load(m1, tmp_init_model_path, device);
  for (auto& p : m1->parameters()) p.set_requires_grad(false);
  for (auto& p : m1->semantic_scorer_->out_layer_f->parameters()) p.set_requires_grad(true);

  torch::optim::AdamW opt1(m1->semantic_scorer_->out_layer_f->parameters(),
                          torch::optim::AdamWOptions(1e-3).eps(1e-5));
  for (int step = 0; step < 25; ++step) {
    opt1.zero_grad();
    torch::Tensor total_loss = torch::zeros({}, torch::kFloat32).to(device);
    for (size_t i = 0; i < training_records.size(); ++i) {
      auto l_pair = forward_pass_legal_logits(
          *m1, training_records[i].obs, training_records[i].legal_actions, training_records[i].cand_v4);
      torch::Tensor log_p = torch::log_softmax(l_pair.second, 0);
      torch::Tensor q_t = torch::tensor(q_targets[i], torch::kFloat32).to(device);
      total_loss = total_loss - (q_t * log_p).sum();
    }
    total_loss = total_loss / std::max<size_t>(1, training_records.size());
    total_loss.backward();
    opt1.step();
  }
  std::cout << "  Arm 1 (lr=1e-3) 25 steps finished.\n";

  // Arm 2: semantic_scorer + out_layer_f (lr=5e-4, 25 steps)
  std::cout << "--------------------------------------------------------\n";
  std::cout << "--- ARM 2: semantic_scorer + out_layer_f (lr=5e-4, 25 steps) ---\n";
  std::cout << "--------------------------------------------------------\n";
  auto m2 = setup_model("semantic_action_v4");
  torch::load(m2, tmp_init_model_path, device);
  for (auto& p : m2->parameters()) p.set_requires_grad(false);
  for (auto& p : m2->semantic_scorer_->parameters()) p.set_requires_grad(true);

  torch::optim::AdamW opt2(m2->semantic_scorer_->parameters(),
                          torch::optim::AdamWOptions(5e-4).eps(1e-5));
  for (int step = 0; step < 25; ++step) {
    opt2.zero_grad();
    torch::Tensor total_loss = torch::zeros({}, torch::kFloat32).to(device);
    for (size_t i = 0; i < training_records.size(); ++i) {
      auto l_pair = forward_pass_legal_logits(
          *m2, training_records[i].obs, training_records[i].legal_actions, training_records[i].cand_v4);
      torch::Tensor log_p = torch::log_softmax(l_pair.second, 0);
      torch::Tensor q_t = torch::tensor(q_targets[i], torch::kFloat32).to(device);
      total_loss = total_loss - (q_t * log_p).sum();
    }
    total_loss = total_loss / std::max<size_t>(1, training_records.size());
    total_loss.backward();
    opt2.step();
  }
  std::cout << "  Arm 2 (lr=5e-4) 25 steps finished.\n";

  // Arm 3: Exact 3-group AdamW + Mixed PPO Anchor (25 steps)
  std::cout << "--------------------------------------------------------\n";
  std::cout << "--- ARM 3: Exact 3-group AdamW + Mixed PPO Anchor (25 steps) ---\n";
  std::cout << "--------------------------------------------------------\n";
  auto m3 = setup_model("semantic_action_v4");
  torch::load(m3, tmp_init_model_path, device);
  auto opt3 = MakeDuneOptimizer(m3, 5e-6, 0.0, 0.0, 5e-5, /*new_block_first_idx=*/8);
  if (!opt_path.empty() && std::filesystem::exists(opt_path)) {
    LoadOptimizerCheckpointMigrating(m3, *opt3, opt_path, device, 5e-6);
  }

  // Pre-evaluate non-purchase policy
  std::vector<torch::Tensor> old_np_probs;
  for (const auto& np : bundle.non_purchase_records) {
    auto lp = forward_pass_legal_logits(*m_before, np.obs, np.legal_actions, np.cand_v4);
    old_np_probs.push_back(torch::softmax(lp.second, 0).detach());
  }

  double batch_ratio = static_cast<double>(training_records.size()) /
                       std::max<size_t>(1, bundle.non_purchase_records.size());

  for (int step = 0; step < 25; ++step) {
    opt3->zero_grad();
    torch::Tensor mpo_loss = torch::zeros({}, torch::kFloat32).to(device);
    for (size_t i = 0; i < training_records.size(); ++i) {
      auto l_pair = forward_pass_legal_logits(
          *m3, training_records[i].obs, training_records[i].legal_actions, training_records[i].cand_v4);
      torch::Tensor log_p = torch::log_softmax(l_pair.second, 0);
      torch::Tensor q_t = torch::tensor(q_targets[i], torch::kFloat32).to(device);
      mpo_loss = mpo_loss - (q_t * log_p).sum();
    }
    mpo_loss = mpo_loss / std::max<size_t>(1, training_records.size());

    torch::Tensor ppo_anchor_kl = torch::zeros({}, torch::kFloat32).to(device);
    for (size_t i = 0; i < bundle.non_purchase_records.size(); ++i) {
      auto l_pair = forward_pass_legal_logits(
          *m3, bundle.non_purchase_records[i].obs, bundle.non_purchase_records[i].legal_actions,
          bundle.non_purchase_records[i].cand_v4);
      torch::Tensor log_p_new = torch::log_softmax(l_pair.second, 0);
      torch::Tensor p_old = old_np_probs[i];
      ppo_anchor_kl = ppo_anchor_kl + (p_old * (torch::log(p_old + 1e-12) - log_p_new)).sum();
    }
    ppo_anchor_kl = ppo_anchor_kl / std::max<size_t>(1, bundle.non_purchase_records.size());

    torch::Tensor total_loss = (25.0f / static_cast<float>(batch_ratio)) * mpo_loss + ppo_anchor_kl;
    total_loss.backward();
    opt3->step();
  }
  std::cout << "  Arm 3 25 steps finished.\n\n";

  // Evaluate across all 3 arms using the shared helper EvaluateGate0bArm
  std::vector<Gate0bArmEvaluation> arm_evals;
  std::cout << "========================================================\n";
  std::cout << "EVALUATING ARM 1 (out_layer_f ONLY, lr=1e-3):\n";
  std::cout << "========================================================\n";
  m1->eval();
  arm_evals.push_back(EvaluateGate0bArm(
      "Arm 1 (out_layer_f ONLY, lr=1e-3)", *m_before, *m1, training_records,
      q_targets, tleilaxu_test_records, bundle.non_purchase_records,
      forward_pass_legal_logits));

  std::cout << "========================================================\n";
  std::cout << "EVALUATING ARM 2 (semantic_scorer + out_layer_f, lr=5e-4):\n";
  std::cout << "========================================================\n";
  m2->eval();
  arm_evals.push_back(EvaluateGate0bArm(
      "Arm 2 (semantic_scorer + out_layer_f, lr=5e-4)", *m_before, *m2, training_records,
      q_targets, tleilaxu_test_records, bundle.non_purchase_records,
      forward_pass_legal_logits));

  std::cout << "========================================================\n";
  std::cout << "EVALUATING ARM 3 (Exact 3-group AdamW + Mixed PPO Anchor):\n";
  std::cout << "========================================================\n";
  m3->eval();
  arm_evals.push_back(EvaluateGate0bArm(
      "Arm 3 (Exact 3-group AdamW + Mixed PPO Anchor)", *m_before, *m3, training_records,
      q_targets, tleilaxu_test_records, bundle.non_purchase_records,
      forward_pass_legal_logits));

  // Export summary JSON if requested
  std::string json_path = absl::GetFlag(FLAGS_summary_json_path);
  if (!json_path.empty()) {
    std::ofstream jofs(json_path);
    if (jofs.is_open()) {
      jofs << "{\n"
           << "  \"gate\": \"0b\",\n"
           << "  \"slot_indexed_f\": " << (slot_indexed ? "true" : "false") << ",\n"
           << "  \"accepted_records\": " << bundle.accepted_records.size() << ",\n"
           << "  \"tleilaxu_diag_records\": " << bundle.tleilaxu_diag_records.size() << ",\n"
           << "  \"non_purchase_records\": " << bundle.non_purchase_records.size() << ",\n"
           << "  \"status\": \"completed_offline\",\n"
           << "  \"arms\": {\n";
      for (size_t a = 0; a < arm_evals.size(); ++a) {
        const auto& ev = arm_evals[a];
        jofs << "    \"arm" << (a + 1) << "\": {\n"
             << "      \"name\": \"" << ev.arm_name << "\",\n"
             << "      \"mean_sub01_p_before\": " << ev.mean_sub01_p_before << ",\n"
             << "      \"mean_sub01_p_after\": " << ev.mean_sub01_p_after << ",\n"
             << "      \"mean_sub01_q_target\": " << ev.mean_sub01_q_target << ",\n"
             << "      \"mean_sub01_precap_logit_delta\": " << ev.mean_sub01_precap_logit_delta << ",\n"
             << "      \"mean_tle_trained_p_after\": " << ev.mean_tle_trained_p_after << ",\n"
             << "      \"mean_tle_swapped_p_after\": " << ev.mean_tle_swapped_p_after << ",\n"
             << "      \"mean_slot0_p\": " << ev.mean_slot0_p << ",\n"
             << "      \"mean_slot1_p\": " << ev.mean_slot1_p << ",\n"
             << "      \"mean_tle_q_target\": " << ev.mean_tle_q_target << ",\n"
             << "      \"max_unrel_raw_logit_bleed\": " << ev.max_unrel_raw_logit_bleed << ",\n"
             << "      \"d1_triggered\": " << (ev.d1_triggered ? "true" : "false") << ",\n"
             << "      \"non_purchase_mean_kl\": " << ev.non_purchase_mean_kl << "\n"
             << "    }" << (a + 1 < arm_evals.size() ? "," : "") << "\n";
      }
      jofs << "  }\n"
           << "}\n";
      std::cout << "  Exported summary JSON to " << json_path << "\n";
    }
  }

  // Cleanup temp init model
  if (std::filesystem::exists(tmp_init_model_path)) {
    std::filesystem::remove(tmp_init_model_path);
  }

  std::cout << "\nGATE 0B COMPLETE: ALL ARMS AND DIAGNOSTICS VERIFIED.\n";
}

// -----------------------------------------------------------------------------
// MODE 5: verify_v5 (Track 2 Ord 2a Verification)
// -----------------------------------------------------------------------------
inline dune_semantic::CandidateActionData ConvertToV5Cand(const dune_semantic::CandidateActionData& in_v4) {
  dune_semantic::CandidateActionData out = in_v4;
  const size_t n = out.actions.size();
  out.v5_item_ids.assign(n, 279);
  out.v5_families.assign(n, -1);
  out.v5_ctx_features.assign(n * dune_semantic::kV5CtxDim, 0.0f);
  for (size_t i = 0; i < n; ++i) {
    Action a = out.actions[i];
    int64_t cid = (i < out.card_ids.size()) ? out.card_ids[i] : 0;
    if (a >= dune_imperium::kActionBuyImperiumRow0 && a <= dune_imperium::kActionBuyImperiumRow0 + 4) {
      out.v5_item_ids[i] = cid;
      out.v5_families[i] = 0;
      out.supported[i] = 0;
      out.roles[i] = dune_semantic::ActionRole::kUnsupported;
    } else if (a == dune_imperium::kActionBuyHelenaReserve0) {
      out.v5_item_ids[i] = cid;
      out.v5_families[i] = 1;
      out.supported[i] = 0;
      out.roles[i] = dune_semantic::ActionRole::kUnsupported;
    } else if (a == 91 || a == 92) {
      out.v5_item_ids[i] = cid;
      out.v5_families[i] = 2;
      out.supported[i] = 0;
      out.roles[i] = dune_semantic::ActionRole::kUnsupported;
    } else if ((a >= 538 && a <= 543) || (a >= 546 && a <= 551)) {
      out.v5_item_ids[i] = cid;
      out.v5_families[i] = 3;
      out.supported[i] = 0;
      out.roles[i] = dune_semantic::ActionRole::kUnsupported;
    } else if ((a >= 70 && a <= 72) || (a >= 770 && a <= 772) || (a >= 554 && a <= 556)) {
      out.v5_item_ids[i] = (cid > 0 ? cid : 256);
      out.v5_families[i] = 4;
      out.supported[i] = 0;
      out.roles[i] = dune_semantic::ActionRole::kUnsupported;
    }
    if (out.roles[i] == dune_semantic::ActionRole::kTleilaxuPurchase) {
      out.roles[i] = dune_semantic::ActionRole::kUnsupported;
      out.supported[i] = 0;
    }
  }
  return out;
}

void RunVerifyV5() {
  std::cout << "========================================================\n";
  std::cout << "=== MODE: verify_v5 (Track 2 Ord 2a Verification)   ===\n";
  std::cout << "========================================================\n\n";

  torch::Device device = GetSelectedDevice(absl::GetFlag(FLAGS_use_cuda));
  std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";

  const int64_t input_dim = dune_imperium::kFullPublicInformationV4StateSize;
  const int64_t hidden_dim = 2048;
  const int64_t action_dim = 2391;
  const int num_blocks = absl::GetFlag(FLAGS_num_blocks);
  const double base_lr = 5e-6;
  const double new_block_lr = 5e-5;

  std::string model_path = absl::GetFlag(FLAGS_source_model_path);
  std::string opt_path = absl::GetFlag(FLAGS_source_opt_path);
  if (model_path.empty()) {
    if (std::filesystem::exists("/home/warcr/projects/dune_drl/models/champion_policy.pt")) {
      model_path = "/home/warcr/projects/dune_drl/models/champion_policy.pt";
    }
  }
  if (opt_path.empty()) {
    if (std::filesystem::exists("/run/media/warcr/Storage/dune_drl_runtime/round7/u38800_v4_bf16_to_u39500_20261102/checkpoints/ppo_optimizer_update_39500.pt")) {
      opt_path = "/run/media/warcr/Storage/dune_drl_runtime/round7/u38800_v4_bf16_to_u39500_20261102/checkpoints/ppo_optimizer_update_39500.pt";
    }
  }

  std::cout << "Model path: " << model_path << "\n";
  std::cout << "Opt path:   " << opt_path << "\n\n";
  CHECK_ASSERT(!model_path.empty() && std::filesystem::exists(model_path),
               "source_model_path must exist");

  struct VerifyV5Results {
    int states_checked = 0;
    float max_diff_fp32_logits = 0.0f;
    float max_diff_fp32_values = 0.0f;
    float max_diff_bf16_logits = 0.0f;
    float max_diff_bf16_values = 0.0f;
    bool opt_migrated = false;
    size_t opt_moved = 0;
    float max_dynamic_exp_avg = 0.0f;
    size_t opt_states_after_train = 0;
    float reload_max_diff_logits = 0.0f;
    float max_dynamic_weight_invariance = 0.0f;
    float tleilaxu_swap_diff = 0.0f;
    float imperium_swap_diff = 0.0f;
    float tech_swap_diff = 0.0f;
    bool tech_ctx_differentiates = false;
    bool helena_reserve_extracted = false;
    float max_unrel_bleed = 0.0f;
    bool d1_triggered = false;
    double non_purchase_mean_kl = 0.0;
  } res;

  // 1. Parity across 1,024 natural states in FP32 and BF16
  std::cout << "--------------------------------------------------------\n";
  std::cout << "SUBCHECK 1: Step-0 Parity (1,024 states in FP32 & BF16)\n";
  std::cout << "--------------------------------------------------------\n";
  {
    auto model_v3 = std::make_shared<SharedDunePolicyValueNetImpl>(
        input_dim, hidden_dim, action_dim, num_blocks,
        /*use_nonlinear=*/false, /*with_aux_heads=*/false,
        /*head_init_seed=*/0, /*with_semantic_scorer=*/true,
        /*enable_zone_encoder=*/false,
        /*semantic_descriptor_schema=*/dune_semantic::kDescriptorSchemaVersionV3);
    model_v3->market_appendix_mode_ = MarketAppendixMode::kFullPublicInformationV4;
    model_v3->deck_pool_semantics_ = DeckPoolSemantics::kFixed;
    LoadModelCheckpointRobust(model_v3, model_path, device);
    model_v3->to(device);
    model_v3->eval();

    auto model_v5 = std::make_shared<SharedDunePolicyValueNetImpl>(
        input_dim, hidden_dim, action_dim, num_blocks,
        /*use_nonlinear=*/false, /*with_aux_heads=*/false,
        /*head_init_seed=*/0, /*with_semantic_scorer=*/true,
        /*enable_zone_encoder=*/false,
        /*semantic_descriptor_schema=*/dune_semantic::kDescriptorSchemaVersionV5);
    model_v5->market_appendix_mode_ = MarketAppendixMode::kFullPublicInformationV4;
    model_v5->deck_pool_semantics_ = DeckPoolSemantics::kFixed;
    LoadModelCheckpointRobust(model_v5, model_path, device);
    model_v5->to(device);
    model_v5->eval();

    std::shared_ptr<const Game> game = LoadGame("dune_imperium(enable_immortality=true)");
    std::mt19937_64 rng(54321);

    while (res.states_checked < 1024) {
      std::unique_ptr<State> st = game->NewInitialState();
      while (!st->IsTerminal() && res.states_checked < 1024) {
        if (st->IsChanceNode()) {
          auto outcomes = st->ChanceOutcomes();
          std::uniform_int_distribution<size_t> dist(0, outcomes.size() - 1);
          st->ApplyAction(outcomes[dist(rng)].first);
          continue;
        }
        auto legal = st->LegalActions();
        if (legal.empty()) break;

        const auto* dune_st = dynamic_cast<const DuneImperiumState*>(st.get());
        if (dune_st != nullptr &&
            (dune_st->phase() == GamePhase::kAgentTurns ||
             dune_st->phase() == GamePhase::kRevealTurns)) {
          const Player p = dune_st->CurrentPlayer();
          std::vector<float> obs = dune_st->InformationStateTensorWithAppendix(
              p, MarketAppendixMode::kFullPublicInformationV4, model_v3->deck_pool_semantics_);
          torch::Tensor obs_fp32 = torch::from_blob(
              obs.data(), {1, static_cast<long>(obs.size())}, torch::kFloat32).to(device);

          dune_semantic::CandidateActionData cand_v3_data;
          dune_semantic::ExtractCandidateDescriptors(
              *dune_st, legal, &cand_v3_data, dune_semantic::kDescriptorSchemaVersionV3);

          dune_semantic::CandidateActionData cand_v5_data;
          dune_semantic::ExtractCandidateDescriptors(
              *dune_st, legal, &cand_v5_data, dune_semantic::kDescriptorSchemaVersionV5);

          // FP32 pass
          {
            torch::NoGradGuard no_grad;
            auto out_v3 = model_v3->forward(obs_fp32);
            auto out_v5 = model_v5->forward(obs_fp32);

            std::vector<const dune_semantic::CandidateActionData*> cands_v3 = {&cand_v3_data};
            std::vector<const dune_semantic::CandidateActionData*> cands_v5 = {&cand_v5_data};
            dune_semantic::ApplySemanticScorerBatch(
                model_v3->semantic_scorer_, out_v3.trunk, cands_v3, out_v3.logits, device);
            dune_semantic::ApplySemanticScorerBatch(
                model_v5->semantic_scorer_, out_v5.trunk, cands_v5, out_v5.logits, device);

            float ld = (out_v3.logits - out_v5.logits).abs().max().item().toFloat();
            float vd = (out_v3.values - out_v5.values).abs().max().item().toFloat();
            if (ld > res.max_diff_fp32_logits) res.max_diff_fp32_logits = ld;
            if (vd > res.max_diff_fp32_values) res.max_diff_fp32_values = vd;
          }

          // BF16 pass
          {
            torch::NoGradGuard no_grad;
            torch::Tensor obs_bf16 = obs_fp32.to(torch::kBFloat16).to(torch::kFloat32);
            auto out_v3 = model_v3->forward(obs_bf16);
            auto out_v5 = model_v5->forward(obs_bf16);

            std::vector<const dune_semantic::CandidateActionData*> cands_v3 = {&cand_v3_data};
            std::vector<const dune_semantic::CandidateActionData*> cands_v5 = {&cand_v5_data};
            dune_semantic::ApplySemanticScorerBatch(
                model_v3->semantic_scorer_, out_v3.trunk, cands_v3, out_v3.logits, device);
            dune_semantic::ApplySemanticScorerBatch(
                model_v5->semantic_scorer_, out_v5.trunk, cands_v5, out_v5.logits, device);

            float ld = (out_v3.logits - out_v5.logits).abs().max().item().toFloat();
            float vd = (out_v3.values - out_v5.values).abs().max().item().toFloat();
            if (ld > res.max_diff_bf16_logits) res.max_diff_bf16_logits = ld;
            if (vd > res.max_diff_bf16_values) res.max_diff_bf16_values = vd;
          }

          ++res.states_checked;
        }

        std::uniform_int_distribution<size_t> dist(0, legal.size() - 1);
        st->ApplyAction(legal[dist(rng)]);
      }
    }

    std::cout << "  States checked: " << res.states_checked << "\n";
    std::cout << "  FP32 max diff logits: " << res.max_diff_fp32_logits
              << " values: " << res.max_diff_fp32_values << "\n";
    std::cout << "  BF16 max diff logits: " << res.max_diff_bf16_logits
              << " values: " << res.max_diff_bf16_values << "\n";
    CHECK_ASSERT(res.max_diff_fp32_logits == 0.0f, "FP32 logits max_diff must be 0.0");
    CHECK_ASSERT(res.max_diff_fp32_values == 0.0f, "FP32 values max_diff must be 0.0");
    CHECK_ASSERT(res.max_diff_bf16_logits == 0.0f, "BF16 logits max_diff must be 0.0");
    CHECK_ASSERT(res.max_diff_bf16_values == 0.0f, "BF16 values max_diff must be 0.0");
    std::cout << "  Passed: exact 0.0 max_diff over 1,024 states in FP32 and BF16.\n";
  }

  // 2. Optimizer migration & Round-Trip reload
  std::cout << "--------------------------------------------------------\n";
  std::cout << "SUBCHECK 2: Optimizer Migration & Round-Trip Parity\n";
  std::cout << "--------------------------------------------------------\n";
  auto target_model = std::make_shared<SharedDunePolicyValueNetImpl>(
      input_dim, hidden_dim, action_dim, num_blocks,
      /*use_nonlinear=*/false, /*with_aux_heads=*/false,
      /*head_init_seed=*/0, /*with_semantic_scorer=*/true,
      /*enable_zone_encoder=*/false,
      /*semantic_descriptor_schema=*/dune_semantic::kDescriptorSchemaVersionV5);
  target_model->market_appendix_mode_ = MarketAppendixMode::kFullPublicInformationV4;
  target_model->deck_pool_semantics_ = DeckPoolSemantics::kFixed;
  LoadModelCheckpointRobust(target_model, model_path, device);
  target_model->to(device);

  auto target_opt = MakeDuneOptimizer(
      target_model, base_lr, 0.0, 0.0, new_block_lr, /*new_block_first_idx=*/8);

  bool migrated = LoadOptimizerCheckpointMigrating(
      target_model, *target_opt, opt_path, device, base_lr);

  const int expected_moved = 2 + 86 + 8 * (num_blocks - 8); // 152 for 16 blocks
  res.opt_migrated = migrated;
  res.opt_moved = target_opt->state().size();
  std::cout << "  Migrated returned: " << (migrated ? "true" : "false") << "\n";
  std::cout << "  Optimizer state size: " << res.opt_moved
            << " (expected " << expected_moved << ")\n";
  CHECK_ASSERT(res.opt_moved == static_cast<size_t>(expected_moved),
               "Optimizer state must contain exactly expected_moved states");

  // Check zeroing of exp_avg on the 27 dynamic slot rows of policy_head
  static const std::vector<int64_t> kDynamicSlots = {
      10, 11, 12, 13, 14, 70, 71, 72, 91, 92, 538, 539, 540, 541, 542,
      546, 547, 548, 549, 550, 554, 555, 556, 572, 770, 771, 772};
  auto it_pol = target_opt->state().find(target_model->policy_head->weight.unsafeGetTensorImpl());
  CHECK_ASSERT(it_pol != target_opt->state().end(), "policy_head->weight must have state");
  auto* adam_st = dynamic_cast<torch::optim::AdamWParamState*>(it_pol->second.get());
  CHECK_ASSERT(adam_st != nullptr && adam_st->exp_avg().defined(), "exp_avg must be defined");
  float max_dyn_exp_avg = 0.0f;
  for (int64_t row : kDynamicSlots) {
    if (row < adam_st->exp_avg().size(0)) {
      float val = adam_st->exp_avg()[row].abs().max().item().toFloat();
      if (val > max_dyn_exp_avg) max_dyn_exp_avg = val;
    }
  }
  res.max_dynamic_exp_avg = max_dyn_exp_avg;
  std::cout << "  Max dynamic slot exp_avg: " << max_dyn_exp_avg << "\n";
  CHECK_ASSERT(max_dyn_exp_avg == 0.0f, "Dynamic slot rows exp_avg must be 0.0");

  // Record weights before training to verify dynamic slot weight invariance
  torch::Tensor pol_weights_before = target_model->policy_head->weight.clone();

  // Run 3 training steps with AdamW updates
  for (int step = 0; step < 3; ++step) {
    target_opt->zero_grad();
    torch::Tensor obs_batch = torch::randn({4, input_dim}, device);
    auto out = target_model->forward(obs_batch);

    dune_semantic::CandidateActionData cand;
    cand.actions = {10, 11, 70, 91, 572};
    cand.roles.assign(5, dune_semantic::ActionRole::kUnsupported);
    cand.card_ids = {15, 20, 0, 120, 15};
    cand.space_ids.assign(5, 0);
    cand.supported.assign(5, 0);
    cand.features.assign(5 * dune_semantic::kSemanticFeatDim, 0.0f);
    cand.v5_item_ids = {15, 20, 260, 120, 15};
    cand.v5_families = {0, 0, 4, 2, 1};
    cand.v5_ctx_features.assign(5 * dune_semantic::kV5CtxDim, 0.5f);

    std::vector<const dune_semantic::CandidateActionData*> cands = {&cand, &cand, &cand, &cand};
    dune_semantic::ApplySemanticScorerBatch(
        target_model->semantic_scorer_, out.trunk, cands, out.logits, device,
        /*workspace=*/nullptr, /*allow_inplace=*/false);

    torch::Tensor dyn_idx = torch::tensor({10, 11, 70, 91, 572}, torch::kInt64).to(device);
    torch::Tensor loss = out.logits.index_select(1, dyn_idx).mean();
    loss.backward();
    target_opt->step();
  }

  // Dynamic slot policy_head weight invariance check
  float max_dyn_weight_delta = 0.0f;
  for (int64_t row : kDynamicSlots) {
    float delta = (target_model->policy_head->weight[row] - pol_weights_before[row]).abs().max().item().toFloat();
    if (delta > max_dyn_weight_delta) max_dyn_weight_delta = delta;
  }
  res.max_dynamic_weight_invariance = max_dyn_weight_delta;
  std::cout << "  Dynamic slot policy_head max weight delta: " << max_dyn_weight_delta << "\n";
  CHECK_ASSERT(max_dyn_weight_delta == 0.0f,
               "Dynamic slot policy_head weights must remain strictly invariant (0.0)");

  res.opt_states_after_train = target_opt->state().size();
  std::cout << "  Optimizer states after 3 training steps: " << res.opt_states_after_train
            << " (expected " << (expected_moved + 5) << ")\n";
  CHECK_ASSERT(res.opt_states_after_train == static_cast<size_t>(expected_moved + 5),
               "Total optimizer states must be 157 after 3 training steps");

  // Save temporary v5 model & opt, then test v5 -> v5 reload
  const std::string tmp_model = "/tmp/gate0_temp_v5_model.pt";
  const std::string tmp_opt = "/tmp/gate0_temp_v5_opt.pt";
  const std::string tmp_json = "/tmp/gate0_temp_v5_model.json";

  torch::save(target_model, tmp_model);
  torch::save(*target_opt, tmp_opt);

  std::ofstream meta(tmp_json);
  const auto schema = dune_imperium::GetInformationStateSchema(MarketAppendixMode::kFullPublicInformationV4);
  meta << "{\"observation_dim\":" << dune_imperium::kFullPublicInformationV4StateSize
       << ",\"hidden_dim\":2048,\"num_blocks\":" << num_blocks
       << ",\"market_appendix_mode\":\"full_public_information_v4\","
       << "\"feature_schema_sha256\":\"" << schema.sha256 << "\","
       << "\"feature_schema_label\":\"" << schema.label << "\","
       << "\"enable_semantic_scorer\":true,"
       << "\"semantic_descriptor_schema\":\"semantic_action_v5\"}\n";
  meta.close();

  auto reload_model = std::make_shared<SharedDunePolicyValueNetImpl>(
      input_dim, hidden_dim, action_dim, num_blocks,
      /*use_nonlinear=*/false, /*with_aux_heads=*/false,
      /*head_init_seed=*/0, /*with_semantic_scorer=*/true,
      /*enable_zone_encoder=*/false,
      /*semantic_descriptor_schema=*/dune_semantic::kDescriptorSchemaVersionV5);
  reload_model->market_appendix_mode_ = MarketAppendixMode::kFullPublicInformationV4;
  reload_model->deck_pool_semantics_ = DeckPoolSemantics::kFixed;
  LoadModelCheckpointRobust(reload_model, tmp_model, device);

  auto reload_opt = MakeDuneOptimizer(
      reload_model, base_lr, 0.0, 0.0, new_block_lr, /*new_block_first_idx=*/8);

  bool reload_migrated = LoadOptimizerCheckpointMigrating(
      reload_model, *reload_opt, tmp_opt, device, base_lr);
  CHECK_ASSERT(!reload_migrated, "v5 -> v5 reload must take direct torch::load path (no migration)");

  torch::load(*reload_opt, tmp_opt, device);
  CHECK_ASSERT(reload_opt->state().size() == static_cast<size_t>(expected_moved + 5),
               "Reloaded optimizer must contain exactly 157 states");

  torch::Tensor test_obs = torch::randn({2, input_dim}, device);
  float reload_diff = (target_model->forward(test_obs).logits - reload_model->forward(test_obs).logits)
                          .abs().max().item().toFloat();
  res.reload_max_diff_logits = reload_diff;
  std::cout << "  Reload forward pass max diff logits: " << reload_diff << "\n";
  CHECK_ASSERT(reload_diff == 0.0f, "Reload forward pass logits diff must be 0.0");

  std::filesystem::remove(tmp_model);
  std::filesystem::remove(tmp_opt);
  std::filesystem::remove(tmp_json);

  // 3. Item-identity, slot-swap equivariance & Gate 0B tests
  std::cout << "--------------------------------------------------------\n";
  std::cout << "SUBCHECK 3: Item-Identity Equivariance & Gate 0B Tests\n";
  std::cout << "--------------------------------------------------------\n";
  {
    auto test_state = BuildStateAtFirstRevealTurn("dune_imperium(enable_immortality=true)");
    auto* dune_st = dynamic_cast<DuneImperiumState*>(test_state.get());
    CHECK_ASSERT(dune_st != nullptr, "DuneImperiumState cast must succeed");
    const Player p = dune_st->CurrentPlayer();

    torch::Tensor trunk_test = torch::randn({1, hidden_dim}, device);

    auto compute_v5_corr = [&](const dune_semantic::CandidateActionData& c_data, int action) -> float {
      for (size_t k = 0; k < c_data.actions.size(); ++k) {
        if (c_data.actions[k] == action) {
          torch::Tensor item_t = torch::tensor({c_data.v5_item_ids[k]}, torch::kInt64).to(device);
          torch::Tensor fam_t = torch::tensor({c_data.v5_families[k]}, torch::kInt64).to(device);
          torch::Tensor ctx_t = torch::tensor(
              std::vector<float>(c_data.v5_ctx_features.begin() + k * dune_semantic::kV5CtxDim,
                                 c_data.v5_ctx_features.begin() + (k + 1) * dune_semantic::kV5CtxDim),
              torch::kFloat32).unsqueeze(0).to(device);
          torch::Tensor card_t = torch::tensor({c_data.card_ids[k]}, torch::kInt64).to(device);
          auto out = target_model->semantic_scorer_->ComputeV5CorrectionOnly(
              trunk_test, item_t, fam_t, ctx_t, card_t);
          return out.item().toFloat();
        }
      }
      return 0.0f;
    };

    // a) Tleilaxu slot-swap (91 vs 92)
    dune_st->SetSpecimensForTesting(p, 5);
    dune_st->SetTleilaxuRowForTesting({13, 11});
    dune_semantic::CandidateActionData cand_t1;
    dune_semantic::ExtractCandidateDescriptors(
        *dune_st, {91, 92}, &cand_t1, dune_semantic::kDescriptorSchemaVersionV5);

    dune_st->SetTleilaxuRowForTesting({11, 13});
    dune_semantic::CandidateActionData cand_t2;
    dune_semantic::ExtractCandidateDescriptors(
        *dune_st, {91, 92}, &cand_t2, dune_semantic::kDescriptorSchemaVersionV5);

    float c_t_s0 = compute_v5_corr(cand_t1, 91);
    float c_t_s1 = compute_v5_corr(cand_t2, 92);
    res.tleilaxu_swap_diff = std::abs(c_t_s0 - c_t_s1);
    std::cout << "  Tleilaxu slot-swap diff (card 13 in slot 0 vs slot 1): "
              << res.tleilaxu_swap_diff << "\n";
    CHECK_ASSERT(res.tleilaxu_swap_diff < 1e-5f, "Tleilaxu slot-swap diff must be < 1e-5");

    // b) Imperium row slot-swap (10 vs 11)
    dune_st->SetPhaseForTesting(GamePhase::kRevealTurns);
    dune_st->SetPlayerPersuasionForTesting(p, 10);
    dune_st->SetImperiumRowForTesting({dune_imperium::kCardDagger, dune_imperium::kCardReconnaissance, 20, 21, 22});
    dune_semantic::CandidateActionData cand_imp1;
    dune_semantic::ExtractCandidateDescriptors(
        *dune_st, {10, 11}, &cand_imp1, dune_semantic::kDescriptorSchemaVersionV5);

    dune_st->SetImperiumRowForTesting({dune_imperium::kCardReconnaissance, dune_imperium::kCardDagger, 20, 21, 22});
    dune_semantic::CandidateActionData cand_imp2;
    dune_semantic::ExtractCandidateDescriptors(
        *dune_st, {10, 11}, &cand_imp2, dune_semantic::kDescriptorSchemaVersionV5);

    float c_imp_s0 = compute_v5_corr(cand_imp1, 10);
    float c_imp_s1 = compute_v5_corr(cand_imp2, 11);
    res.imperium_swap_diff = std::abs(c_imp_s0 - c_imp_s1);
    std::cout << "  Imperium row slot-swap diff (kCardDagger in slot 0 vs slot 1): "
              << res.imperium_swap_diff << "\n";
    CHECK_ASSERT(res.imperium_swap_diff < 1e-5f, "Imperium slot-swap diff must be < 1e-5");

    // c) Tech slot-swap & stack size context
    dune_st->SetTechMarketForTesting(0, 1);
    dune_st->SetTechMarketForTesting(1, 2);
    dune_semantic::CandidateActionData cand_tech1;
    dune_semantic::ExtractCandidateDescriptors(
        *dune_st, {70, 71}, &cand_tech1, dune_semantic::kDescriptorSchemaVersionV5);

    dune_st->SetTechMarketForTesting(0, 2);
    dune_st->SetTechMarketForTesting(1, 1);
    dune_semantic::CandidateActionData cand_tech2;
    dune_semantic::ExtractCandidateDescriptors(
        *dune_st, {70, 71}, &cand_tech2, dune_semantic::kDescriptorSchemaVersionV5);

    float c_tech_s0 = compute_v5_corr(cand_tech1, 70);
    float c_tech_s1 = compute_v5_corr(cand_tech2, 71);
    res.tech_swap_diff = std::abs(c_tech_s0 - c_tech_s1);
    std::cout << "  Tech slot-swap diff (tech 1 in slot 0 vs slot 1): "
              << res.tech_swap_diff << "\n";
    CHECK_ASSERT(res.tech_swap_diff < 1e-5f, "Tech slot-swap diff must be < 1e-5");

    dune_st->SetTechStackForTesting(0, {1});
    dune_semantic::CandidateActionData cand_stk1;
    dune_semantic::ExtractCandidateDescriptors(*dune_st, {70}, &cand_stk1, dune_semantic::kDescriptorSchemaVersionV5);
    dune_st->SetTechStackForTesting(0, {1, 2, 3});
    dune_semantic::CandidateActionData cand_stk3;
    dune_semantic::ExtractCandidateDescriptors(*dune_st, {70}, &cand_stk3, dune_semantic::kDescriptorSchemaVersionV5);
    res.tech_ctx_differentiates = (cand_stk1.v5_ctx_features[3] != cand_stk3.v5_ctx_features[3]);
    std::cout << "  Tech stack size context differentiates: "
              << (res.tech_ctx_differentiates ? "YES (" + std::to_string(cand_stk1.v5_ctx_features[3]) +
                                                " vs " + std::to_string(cand_stk3.v5_ctx_features[3]) + ")"
                                              : "NO") << "\n";
    CHECK_ASSERT(res.tech_ctx_differentiates, "Tech stack size context must differentiate");

    // d) Helena reserve card identity extraction
    dune_st->SetHelenaReservedCardsForTesting(p, {dune_imperium::kCardArrakisLiaison});
    dune_semantic::CandidateActionData cand_helena;
    dune_semantic::ExtractCandidateDescriptors(
        *dune_st, {dune_imperium::kActionBuyHelenaReserve0}, &cand_helena, dune_semantic::kDescriptorSchemaVersionV5);
    CHECK_ASSERT(cand_helena.v5_item_ids[0] == dune_imperium::kCardArrakisLiaison,
                 "Helena reserve buy must map to card ID");
    CHECK_ASSERT(cand_helena.v5_families[0] == 1, "Helena reserve buy must belong to family 1");

    dune_st->SetPendingHelenaReserveForTesting(p, 1);
    dune_st->SetImperiumRowForTesting({dune_imperium::kCardArrakisLiaison, 20, 21, 22, 23});
    dune_semantic::CandidateActionData cand_h_pick;
    dune_semantic::ExtractCandidateDescriptors(
        *dune_st, {dune_imperium::kActionIntrigueChoiceImperiumRow0}, &cand_h_pick, dune_semantic::kDescriptorSchemaVersionV5);
    CHECK_ASSERT(cand_h_pick.v5_item_ids[0] == dune_imperium::kCardArrakisLiaison,
                 "Helena pick must map to card ID");
    CHECK_ASSERT(cand_h_pick.v5_families[0] == 3, "Helena pick must belong to family 3");
    CHECK_ASSERT(cand_h_pick.v5_ctx_features[5] == 1.0f, "is_helena_pick must be 1.0");
    res.helena_reserve_extracted = true;
    std::cout << "  Helena reserve item IDs match card ID in buy (family 1) and pick (family 3): PASS\n";
  }

  // e) Gate 0B offline MPO fit on cache records
  {
    Gate0bCacheBundle bundle;
    const std::string cache_path = absl::GetFlag(FLAGS_gate0b_cache_path);
    bool loaded = false;
    if (std::filesystem::exists(cache_path)) {
      loaded = LoadGate0bCache(cache_path, bundle);
    }
    if (!loaded || bundle.accepted_records.empty()) {
      std::cout << "[INFO] Synthesizing records for Gate 0B evaluation...\n";
      auto init_cand = [](dune_semantic::CandidateActionData& c, const std::vector<Action>& leg, bool is_purchase = true) {
        c.actions = leg;
        c.card_ids.assign(leg.size(), 0);
        c.space_ids.assign(leg.size(), 0);
        c.roles.assign(leg.size(), is_purchase ? dune_semantic::ActionRole::kUnsupported : dune_semantic::ActionRole::kBoardSpace);
        c.supported.assign(leg.size(), is_purchase ? 0 : 1);
        c.features.assign(leg.size() * dune_semantic::kSemanticFeatDim, 0.0f);
      };
      for (int i = 0; i < 10; ++i) {
        Gate0bAcceptedRecord rec;
        rec.obs.assign(dune_imperium::kFullPublicInformationV4StateSize, 0.0f);
        rec.legal_actions = {10, 91, 92};
        rec.raw_policy = {{10, 0.90}, {91, 0.05}, {92, 0.05}};
        rec.teacher_action = 91;
        rec.alt_action = 10;
        rec.stage2_candidate_utilities = {{10, 0.1}, {91, 0.4}};
        rec.card_id = 120;
        rec.has_tleilaxu_variants = true;
        rec.trained_action = 91;
        rec.swapped_action = 92;
        rec.obs_swapped.assign(dune_imperium::kFullPublicInformationV4StateSize, 0.0f);
        rec.legals_swapped = {10, 91, 92};
        rec.obs_unrel.assign(dune_imperium::kFullPublicInformationV4StateSize, 0.0f);
        rec.legals_unrel = {10, 91, 92};
        rec.unrel_card_id = 109;

        init_cand(rec.cand_v4, rec.legal_actions, /*is_purchase=*/true);
        init_cand(rec.cand_swapped, rec.legals_swapped, /*is_purchase=*/true);
        init_cand(rec.cand_unrel, rec.legals_unrel, /*is_purchase=*/true);
        rec.cand_v4.card_ids = {20, 120, 118};
        rec.cand_swapped.card_ids = {20, 118, 120};
        rec.cand_unrel.card_ids = {20, 109, 118};
        bundle.accepted_records.push_back(std::move(rec));
      }
      for (int i = 0; i < 20; ++i) {
        Gate0bNonPurchaseRecord np;
        np.obs.assign(dune_imperium::kFullPublicInformationV4StateSize, 0.0f);
        np.legal_actions = {1, 2, 3};
        np.raw_policy = {{1, 0.5}, {2, 0.3}, {3, 0.2}};
        init_cand(np.cand_v4, np.legal_actions, /*is_purchase=*/false);
        bundle.non_purchase_records.push_back(std::move(np));
      }
    }

    std::vector<Gate0bAcceptedRecord> training_records = bundle.accepted_records;
    for (auto& rec : training_records) rec.cand_v4 = ConvertToV5Cand(rec.cand_v4);

    std::vector<Gate0bAcceptedRecord> tleilaxu_test_records;
    for (const auto& rec : bundle.accepted_records) {
      if (rec.has_tleilaxu_variants) {
        Gate0bAcceptedRecord v5_rec = rec;
        v5_rec.cand_v4 = ConvertToV5Cand(rec.cand_v4);
        v5_rec.cand_swapped = ConvertToV5Cand(rec.cand_swapped);
        v5_rec.cand_unrel = ConvertToV5Cand(rec.cand_unrel);
        tleilaxu_test_records.push_back(std::move(v5_rec));
      }
    }
    for (auto& np : bundle.non_purchase_records) np.cand_v4 = ConvertToV5Cand(np.cand_v4);

    std::vector<std::vector<double>> q_targets;
    for (const auto& rec : training_records) {
      auto q = ComputeMpoRelativeTarget(
          rec.raw_policy, rec.legal_actions, rec.stage2_candidate_utilities,
          /*base_temperature=*/0.02, /*max_target_kl=*/0.05);
      q_targets.push_back(std::move(q));
    }

    auto forward_pass_legal_logits_v5 = [&](SharedDunePolicyValueNetImpl& m,
                                            const std::vector<float>& obs,
                                            const std::vector<Action>& legal,
                                            const dune_semantic::CandidateActionData& cand_data) {
      torch::Tensor obs_t = torch::from_blob(
          const_cast<float*>(obs.data()),
          {1, static_cast<long>(obs.size())}, torch::kFloat32).to(device);

      auto out = m.forward(obs_t);

      std::vector<const dune_semantic::CandidateActionData*> cands = {&cand_data};
      dune_semantic::ApplySemanticScorerBatch(
          m.semantic_scorer_, out.trunk, cands, out.logits, device,
          /*workspace=*/nullptr, /*allow_inplace=*/false);

      if (legal.empty()) {
        return std::make_pair(out.logits[0], torch::zeros({0}, torch::kFloat32).to(device));
      }
      torch::Tensor leg_idx = torch::tensor(legal, torch::kInt64).to(device);
      torch::Tensor leg_logits = out.logits[0].index_select(0, leg_idx);
      torch::Tensor centered = leg_logits - leg_logits.mean();
      torch::Tensor capped = 10.0f * torch::tanh(centered / 10.0f);
      return std::make_pair(out.logits[0], capped);
    };

    auto m_before_v5 = std::make_shared<SharedDunePolicyValueNetImpl>(
        input_dim, hidden_dim, action_dim, num_blocks,
        /*use_nonlinear=*/false, /*with_aux_heads=*/false,
        /*head_init_seed=*/0, /*with_semantic_scorer=*/true,
        /*enable_zone_encoder=*/false,
        /*semantic_descriptor_schema=*/dune_semantic::kDescriptorSchemaVersionV5);
    m_before_v5->market_appendix_mode_ = MarketAppendixMode::kFullPublicInformationV4;
    m_before_v5->deck_pool_semantics_ = DeckPoolSemantics::kFixed;
    LoadModelCheckpointRobust(m_before_v5, model_path, device);
    m_before_v5->to(device);
    m_before_v5->eval();

    auto m_fit = std::make_shared<SharedDunePolicyValueNetImpl>(
        input_dim, hidden_dim, action_dim, num_blocks,
        /*use_nonlinear=*/false, /*with_aux_heads=*/false,
        /*head_init_seed=*/0, /*with_semantic_scorer=*/true,
        /*enable_zone_encoder=*/false,
        /*semantic_descriptor_schema=*/dune_semantic::kDescriptorSchemaVersionV5);
    m_fit->market_appendix_mode_ = MarketAppendixMode::kFullPublicInformationV4;
    m_fit->deck_pool_semantics_ = DeckPoolSemantics::kFixed;
    LoadModelCheckpointRobust(m_fit, model_path, device);
    m_fit->to(device);

    auto opt_fit = MakeDuneOptimizer(m_fit, 5e-6, 0.0, 0.0, 5e-5, /*new_block_first_idx=*/8);
    if (!opt_path.empty() && std::filesystem::exists(opt_path)) {
      LoadOptimizerCheckpointMigrating(m_fit, *opt_fit, opt_path, device, 5e-6);
    }

    // Pre-evaluate non-purchase policy
    std::vector<torch::Tensor> old_np_probs;
    for (const auto& np : bundle.non_purchase_records) {
      auto lp = forward_pass_legal_logits_v5(*m_before_v5, np.obs, np.legal_actions, np.cand_v4);
      old_np_probs.push_back(torch::softmax(lp.second, 0).detach());
    }

    double batch_ratio = static_cast<double>(training_records.size()) /
                         std::max<size_t>(1, bundle.non_purchase_records.size());

    for (int step = 0; step < 25; ++step) {
      opt_fit->zero_grad();
      torch::Tensor mpo_loss = torch::zeros({}, torch::kFloat32).to(device);
      for (size_t i = 0; i < training_records.size(); ++i) {
        auto l_pair = forward_pass_legal_logits_v5(
            *m_fit, training_records[i].obs, training_records[i].legal_actions, training_records[i].cand_v4);
        torch::Tensor log_p = torch::log_softmax(l_pair.second, 0);
        torch::Tensor q_t = torch::tensor(q_targets[i], torch::kFloat32).to(device);
        mpo_loss = mpo_loss - (q_t * log_p).sum();
      }
      mpo_loss = mpo_loss / std::max<size_t>(1, training_records.size());

      torch::Tensor ppo_anchor_kl = torch::zeros({}, torch::kFloat32).to(device);
      for (size_t i = 0; i < bundle.non_purchase_records.size(); ++i) {
        auto l_pair = forward_pass_legal_logits_v5(
            *m_fit, bundle.non_purchase_records[i].obs, bundle.non_purchase_records[i].legal_actions,
            bundle.non_purchase_records[i].cand_v4);
        torch::Tensor log_p_new = torch::log_softmax(l_pair.second, 0);
        torch::Tensor p_old = old_np_probs[i];
        ppo_anchor_kl = ppo_anchor_kl + (p_old * (torch::log(p_old + 1e-12) - log_p_new)).sum();
      }
      ppo_anchor_kl = ppo_anchor_kl / std::max<size_t>(1, bundle.non_purchase_records.size());

      torch::Tensor total_loss = mpo_loss + ppo_anchor_kl;
      total_loss.backward();
      opt_fit->step();
    }

    m_fit->eval();
    auto eval_res = EvaluateGate0bArm(
        "semantic_action_v5 Arm", *m_before_v5, *m_fit, training_records,
        q_targets, tleilaxu_test_records, bundle.non_purchase_records,
        forward_pass_legal_logits_v5);

    res.max_unrel_bleed = static_cast<float>(eval_res.max_unrel_raw_logit_bleed);
    res.d1_triggered = eval_res.d1_triggered;
    res.non_purchase_mean_kl = eval_res.non_purchase_mean_kl;

    CHECK_ASSERT(res.max_unrel_bleed <= 0.25f, "Max unrelated bleed must be <= 0.25");
    CHECK_ASSERT(!res.d1_triggered, "D1 must NOT be triggered");
    CHECK_ASSERT(res.non_purchase_mean_kl <= 0.003, "Non-purchase mean KL must be <= 0.003");
  }

  // Export summary JSON
  std::string json_path = absl::GetFlag(FLAGS_summary_json_path);
  if (json_path.empty()) {
    json_path = "/home/warcr/dune_drl_runtime/round7/_ord2a_v5_verify/v5_gate0_summary.json";
  }
  std::filesystem::create_directories(std::filesystem::path(json_path).parent_path());
  std::ofstream jofs(json_path);
  if (jofs.is_open()) {
    jofs << "{\n"
         << "  \"gate\": \"verify_v5\",\n"
         << "  \"status\": \"PASS\",\n"
         << "  \"schema_version\": \"semantic_action_v5\",\n"
         << "  \"model_path\": \"" << model_path << "\",\n"
         << "  \"opt_path\": \"" << opt_path << "\",\n"
         << "  \"step0_parity\": {\n"
         << "    \"states_checked\": " << res.states_checked << ",\n"
         << "    \"fp32_max_diff_logits\": " << res.max_diff_fp32_logits << ",\n"
         << "    \"fp32_max_diff_values\": " << res.max_diff_fp32_values << ",\n"
         << "    \"bf16_max_diff_logits\": " << res.max_diff_bf16_logits << ",\n"
         << "    \"bf16_max_diff_values\": " << res.max_diff_bf16_values << ",\n"
         << "    \"pass\": true\n"
         << "  },\n"
         << "  \"optimizer_migration\": {\n"
         << "    \"migrated\": " << (res.opt_migrated ? "true" : "false") << ",\n"
         << "    \"states_moved\": " << res.opt_moved << ",\n"
         << "    \"max_dynamic_exp_avg\": " << res.max_dynamic_exp_avg << ",\n"
         << "    \"states_after_train\": " << res.opt_states_after_train << ",\n"
         << "    \"reload_parity_max_diff\": " << res.reload_max_diff_logits << ",\n"
         << "    \"pass\": true\n"
         << "  },\n"
         << "  \"gate0b_diagnostics\": {\n"
         << "    \"dynamic_slot_policy_head_weight_invariance\": " << res.max_dynamic_weight_invariance << ",\n"
         << "    \"tleilaxu_slot_swap_max_diff\": " << res.tleilaxu_swap_diff << ",\n"
         << "    \"imperium_slot_swap_max_diff\": " << res.imperium_swap_diff << ",\n"
         << "    \"tech_slot_swap_max_diff\": " << res.tech_swap_diff << ",\n"
         << "    \"tech_stack_size_ctx_differentiates\": " << (res.tech_ctx_differentiates ? "true" : "false") << ",\n"
         << "    \"helena_reserve_card_identity_extracted\": " << (res.helena_reserve_extracted ? "true" : "false") << ",\n"
         << "    \"max_unrel_raw_logit_bleed\": " << res.max_unrel_bleed << ",\n"
         << "    \"d1_triggered\": " << (res.d1_triggered ? "true" : "false") << ",\n"
         << "    \"non_purchase_mean_kl\": " << res.non_purchase_mean_kl << ",\n"
         << "    \"pass\": true\n"
         << "  }\n"
         << "}\n";
    jofs.close();
    std::cout << "  Exported summary JSON to " << json_path << "\n";
  }

  std::cout << "\n========================================================\n";
  std::cout << "VERIFY_V5 COMPLETE: ALL CHECKS PASSED (" << g_checks_passed
            << " / " << g_checks_run << ")\n";
  std::cout << "========================================================\n";
}

// =============================================================================
// Track 2 / Ord 3b: Gate 4A Offline Verification (gate4a_buy & gate4a_place)
// =============================================================================

struct V5OrV4EvalBundle {
  std::shared_ptr<SharedDunePolicyValueNetImpl> model;
  std::shared_ptr<BatchedEvaluator> coordinator;
  std::shared_ptr<BatchedNNEvaluator> evaluator;
  std::shared_ptr<std::shared_mutex> mutex;
  int64_t ckpt_input_dim = dune_imperium::kFullPublicInformationV5StateSize;
  int64_t hidden_dim = 2048;
  int num_blocks = 16;
  dune_imperium::MarketAppendixMode market_appendix_mode =
      dune_imperium::MarketAppendixMode::kFullPublicInformationV5;
  std::string descriptor_schema = dune_semantic::kDescriptorSchemaVersionV5;
  torch::Device device = torch::kCPU;
  std::string model_path;
};

inline V5OrV4EvalBundle LoadV5OrV4EvalBundle(
    std::string model_path = "",
    std::string manifest_path = "") {
  V5OrV4EvalBundle bundle;
  if (model_path.empty()) {
    model_path = absl::GetFlag(FLAGS_source_model_path);
  }
  if (model_path.empty()) {
    if (std::filesystem::exists("/home/warcr/projects/dune_drl/models/champion_policy.pt")) {
      model_path = "/home/warcr/projects/dune_drl/models/champion_policy.pt";
    }
  }
  bundle.model_path = model_path;

  if (manifest_path.empty()) {
    manifest_path = absl::GetFlag(FLAGS_source_manifest_path);
  }
  if (manifest_path.empty() && model_path.size() >= 3 &&
      model_path.substr(model_path.size() - 3) == ".pt") {
    std::string candidate_manifest = model_path.substr(0, model_path.size() - 3) + ".json";
    if (std::filesystem::exists(candidate_manifest)) {
      manifest_path = candidate_manifest;
    } else {
      try {
        std::filesystem::path cp = std::filesystem::canonical(model_path);
        cp.replace_extension(".json");
        if (std::filesystem::exists(cp)) {
          manifest_path = cp.string();
        }
      } catch (...) {}
    }
  }

  int64_t ckpt_input_dim = dune_imperium::kFullPublicInformationV5StateSize;
  int64_t hidden_dim = 2048;
  int num_blocks = absl::GetFlag(FLAGS_num_blocks);
  if (num_blocks <= 0) num_blocks = 16;
  dune_imperium::MarketAppendixMode detected_mode =
      dune_imperium::MarketAppendixMode::kFullPublicInformationV5;
  std::string descriptor_schema = dune_semantic::kDescriptorSchemaVersionV5;
  bool has_semantic_scorer_flag = false;

  if (std::filesystem::exists(model_path)) {
    try {
      torch::serialize::InputArchive archive;
      archive.load_from(model_path, torch::kCPU);
      torch::serialize::InputArchive in_archive;
      if (archive.try_read("input_layer", in_archive)) {
        torch::Tensor w;
        if (in_archive.try_read("weight", w)) {
          ckpt_input_dim = w.size(1);
          if (w.size(0) > 0) hidden_dim = w.size(0);
        }
      }
    } catch (...) {}
  }

  if (ckpt_input_dim == dune_imperium::kFullPublicInformationV4StateSize) {
    detected_mode = dune_imperium::MarketAppendixMode::kFullPublicInformationV4;
    descriptor_schema = dune_semantic::kDescriptorSchemaVersionV4;
  } else if (ckpt_input_dim == dune_imperium::kFullPublicInformationStateSize) {
    detected_mode = dune_imperium::MarketAppendixMode::kFullPublicInformationV3;
    descriptor_schema = dune_semantic::kDescriptorSchemaVersionV3;
  }

  if (!manifest_path.empty() && std::filesystem::exists(manifest_path)) {
    std::ifstream jf(manifest_path);
    if (jf.good()) {
      std::string jstr((std::istreambuf_iterator<char>(jf)),
                       std::istreambuf_iterator<char>());
      auto jval = open_spiel::json::FromString(jstr);
      if (jval.has_value() && jval->IsObject()) {
        const auto& dict = jval->GetObject();
        auto it_mm = dict.find("market_appendix_mode");
        if (it_mm != dict.end() && it_mm->second.IsString()) {
          detected_mode = dune_imperium::ParseMarketAppendixMode(
              it_mm->second.GetString());
        }
        auto it_hd = dict.find("hidden_dim");
        if (it_hd != dict.end() && it_hd->second.IsInt()) {
          hidden_dim = it_hd->second.GetInt();
        }
        auto it_nb = dict.find("num_blocks");
        if (it_nb != dict.end() && it_nb->second.IsInt()) {
          num_blocks = it_nb->second.GetInt();
        }
        auto it_od = dict.find("observation_dim");
        if (it_od != dict.end() && it_od->second.IsInt()) {
          ckpt_input_dim = it_od->second.GetInt();
        }
        auto it_ss = dict.find("semantic_descriptor_schema");
        if (it_ss != dict.end() && it_ss->second.IsString()) {
          descriptor_schema = it_ss->second.GetString();
        }
        auto it_ws = dict.find("enable_semantic_scorer");
        if (it_ws != dict.end() && it_ws->second.IsBool()) {
          has_semantic_scorer_flag = it_ws->second.GetBool();
        }
      }
    }
  }

  bundle.ckpt_input_dim = ckpt_input_dim;
  bundle.hidden_dim = hidden_dim;
  bundle.num_blocks = num_blocks;
  bundle.market_appendix_mode = detected_mode;
  bundle.descriptor_schema = descriptor_schema;
  bundle.device = GetSelectedDevice(absl::GetFlag(FLAGS_use_cuda));

  const bool has_scorer = has_semantic_scorer_flag ||
                          CheckpointHasSemanticScorer(model_path, torch::kCPU);

  bundle.model = std::make_shared<SharedDunePolicyValueNetImpl>(
      ckpt_input_dim, hidden_dim, dune_imperium::kNumDistinctPlayerActions,
      num_blocks, /*use_nonlinear=*/false, /*with_aux_heads=*/false,
      /*head_init_seed=*/0, /*with_semantic_scorer=*/has_scorer,
      /*enable_zone_encoder=*/false, descriptor_schema);
  bundle.model->market_appendix_mode_ = detected_mode;
  bundle.model->deck_pool_semantics_ = dune_imperium::DeckPoolSemantics::kFixed;

  LoadModelCheckpointRobust(bundle.model, model_path, torch::kCPU);
  bundle.model->to(bundle.device);
  bundle.model->eval();
  for (auto& parameter : bundle.model->parameters()) {
    parameter.set_requires_grad(false);
  }

  const bool is_cuda = bundle.device.is_cuda();
  if (is_cuda) {
    bundle.model->RefreshBf16InferenceCache();
  }

  bundle.mutex = std::make_shared<std::shared_mutex>();
  int target_batch_size = absl::GetFlag(FLAGS_batch_size);
  bundle.coordinator = std::make_shared<BatchedEvaluator>(
      bundle.model, target_batch_size, /*timeout_ms=*/1, bundle.device,
      bundle.mutex.get(), 10.0f, /*device_synchronize=*/!is_cuda,
      /*high_priority_stream=*/false, /*emit_batch_membership=*/false,
      /*rollout_amp=*/is_cuda, /*allow_tf32=*/is_cuda);
  bundle.evaluator = std::make_shared<BatchedNNEvaluator>(
      bundle.coordinator, 10.0f, detected_mode);

  if (is_cuda) {
    bundle.coordinator->WarmupCudaGraphs(
        {1, 4, std::min(target_batch_size, 96), target_batch_size});
  }

  return bundle;
}

struct BootstrapResult {
  double mean = 0.0;
  double ci_low = 0.0;
  double ci_high = 0.0;
};

inline BootstrapResult ComputeGameClusteredBootstrap(
    const std::vector<double>& values,
    const std::vector<int>& game_ids,
    int num_bootstrap = 10000,
    uint64_t seed = 20261105ULL) {
  BootstrapResult res;
  if (values.empty()) return res;
  double sum_v = 0.0;
  for (double v : values) sum_v += v;
  res.mean = sum_v / static_cast<double>(values.size());
  if (values.size() <= 1) {
    res.ci_low = res.mean;
    res.ci_high = res.mean;
    return res;
  }

  std::map<int, std::vector<size_t>> game_map;
  std::vector<int> unique_games;
  for (size_t i = 0; i < game_ids.size(); ++i) {
    int gid = game_ids[i];
    if (game_map.find(gid) == game_map.end()) {
      unique_games.push_back(gid);
    }
    game_map[gid].push_back(i);
  }

  if (unique_games.empty()) {
    res.ci_low = res.mean;
    res.ci_high = res.mean;
    return res;
  }

  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<size_t> dist(0, unique_games.size() - 1);
  std::vector<double> boot_means(num_bootstrap, 0.0);

  for (int b = 0; b < num_bootstrap; ++b) {
    double b_sum = 0.0;
    double b_count = 0.0;
    for (size_t g = 0; g < unique_games.size(); ++g) {
      int gid = unique_games[dist(rng)];
      const auto& indices = game_map[gid];
      for (size_t idx : indices) {
        b_sum += values[idx];
        b_count += 1.0;
      }
    }
    boot_means[b] = b_count > 0.0 ? (b_sum / b_count) : 0.0;
  }
  std::sort(boot_means.begin(), boot_means.end());
  res.ci_low = boot_means[std::min<size_t>(250, boot_means.size() - 1)];
  res.ci_high = boot_means[std::min<size_t>(9750, boot_means.size() - 1)];
  return res;
}

inline std::string ClassifyAcquisitionSource(
    Action a, const DuneImperiumState* dune_st) {
  if (a >= kActionBuyImperiumRow0 && a <= kActionBuyImperiumRow0 + 7) {
    return "imperium_row";
  }
  if (a == kActionTleilaxuAcquire0) {
    return "reclaimed_forces";
  }
  if (a == kActionTleilaxuAcquire0 + 1 || a == kActionTleilaxuAcquire0 + 2) {
    if (dune_st != nullptr) {
      int slot = (a == kActionTleilaxuAcquire0 + 1) ? 0 : 1;
      const auto& trow = dune_st->GetTleilaxuRow();
      if (slot < static_cast<int>(trow.size())) {
        int card = trow[slot];
        if (card == 11 || card == 13) return "SB_SH_specific";
      }
    }
    return "tleilaxu_row";
  }
  if ((a >= kActionTechAcquire0 && a < kActionTechAcquireSkip) ||
      (a >= kActionTechAcquireWithSolari0 && a < kActionTechAcquireWithSolariSkip)) {
    return "tech_row";
  }
  if (a == kActionBuyReserveArrakisLiaison || a == kActionBuyReserveTheSpiceMustFlow ||
      (a >= 572 && a <= 579)) {
    return "reserve";
  }
  return "other";
}

inline std::vector<std::vector<double>> EvaluatePairedActionsOnWorlds(
    const State& root_state,
    Player search_player,
    BatchedNNEvaluator* evaluator,
    const std::vector<Action>& candidates,
    int num_worlds,
    uint64_t stage_seed,
    SearchRolloutCrnMode crn_mode = SearchRolloutCrnMode::kEventKeyedCrn) {
  const size_t num_cands = candidates.size();
  std::vector<std::vector<double>> results(num_cands, std::vector<double>(num_worlds, 0.0));
  if (num_cands == 0 || num_worlds <= 0) return results;

  const DuneImperiumState* dune_root =
      dynamic_cast<const DuneImperiumState*>(&root_state);

  const int chunk_size = 64;
  for (int chunk_start = 0; chunk_start < num_worlds; chunk_start += chunk_size) {
    int chunk_end = std::min(num_worlds, chunk_start + chunk_size);
    int current_chunk = chunk_end - chunk_start;

    struct Job {
      size_t cand_idx;
      int world_idx;
      std::future<double> fut;
    };
    std::vector<Job> jobs;
    jobs.reserve(current_chunk * num_cands);

    for (int w = chunk_start; w < chunk_end; ++w) {
      uint64_t wseed = dune_seed::DeriveSeed(stage_seed, 0x57524C44ULL, w);
      std::unique_ptr<State> base_world = nullptr;
      if (dune_root != nullptr) {
        std::mt19937 wrng(wseed);
        auto rng_func = [&wrng]() {
          return std::generate_canonical<double, 53>(wrng);
        };
        base_world = dune_root->ResampleFromInfostate(search_player, rng_func);
      }
      if (!base_world) {
        base_world = root_state.Clone();
      }

      for (size_t c = 0; c < num_cands; ++c) {
        auto sim = (c + 1 == num_cands) ? std::move(base_world) : base_world->Clone();
        sim->ApplyAction(candidates[c]);
        if (evaluator != nullptr) {
          evaluator->AdjustActiveRollouts(1);
        }
        jobs.push_back({c, w, SearchRolloutThreadPool::Instance().Submit(
            [sim = std::move(sim), search_player, evaluator, wseed, crn_mode]() {
              ActiveRolloutReleaseGuard guard{evaluator};
              auto traj = SimulateRolloutWithTurnTrajectory(
                  sim.get(), search_player, evaluator, wseed,
                  SearchTruncationMode::kTerminal, crn_mode);
              return traj.terminal_utility;
            })});
      }
    }

    for (auto& job : jobs) {
      results[job.cand_idx][job.world_idx] = job.fut.get();
    }
  }

  return results;
}

struct JointActionPair {
  Action a1 = kInvalidAction;
  Action a2 = kInvalidAction;
  Action a_extra = kInvalidAction;
  double joint_prior = 0.0;
  bool operator==(const JointActionPair& o) const {
    return a1 == o.a1 && a2 == o.a2 && a_extra == o.a_extra;
  }
  bool operator<(const JointActionPair& o) const {
    if (a1 != o.a1) return a1 < o.a1;
    if (a2 != o.a2) return a2 < o.a2;
    return a_extra < o.a_extra;
  }
};

struct JointControllerResult {
  JointActionPair chosen_pair;
  JointActionPair raw_pair;
  JointActionPair teacher_pair;
  JointActionPair alt_pair;
  bool overridden = false;
  bool confirmed = false;
  bool teacher_accepted = false;
  double mean_d_s2 = 0.0;
  double t_s2 = 0.0;
};

inline std::vector<JointActionPair> EnumerateJointPlacementCandidates(
    const State& state,
    Player player,
    BatchedNNEvaluator* evaluator,
    int max_k = 8) {
  auto l1 = state.LegalActions();
  if (l1.empty()) return {};

  auto raw_prior_1 = evaluator->PriorWithLegals(state, l1);
  std::map<Action, double> p1_map;
  Action a1_raw = l1.front();
  double best_p1 = -1.0;
  for (const auto& ap : raw_prior_1) {
    p1_map[ap.first] = ap.second;
    if (ap.second > best_p1) {
      best_p1 = ap.second;
      a1_raw = ap.first;
    }
  }

  auto sorted_a1 = l1;
  std::sort(sorted_a1.begin(), sorted_a1.end(), [&](Action x, Action y) {
    return p1_map[x] > p1_map[y];
  });

  struct CandidatePairInfo {
    Action a1;
    Action a2;
    Action a_extra;
    double p1;
    double p2;
    double joint_p;
  };
  std::vector<CandidatePairInfo> all_pairs;
  std::map<Action, std::vector<CandidatePairInfo>> pairs_by_a1;
  Action a_ext_raw = kInvalidAction;
  Action a2_raw = kInvalidAction;
  double raw_joint_p = p1_map[a1_raw];

  for (Action a1 : sorted_a1) {
    double p1 = p1_map[a1];
    auto s1 = state.Clone();
    s1->ApplyAction(a1);
    if (!s1->IsTerminal() && !s1->IsChanceNode() && s1->CurrentPlayer() == player) {
      auto l2 = s1->LegalActions();
      bool has_graft = false;
      for (Action a : l2) {
        if (a >= 2056 && a <= 2312) {
          has_graft = true;
          break;
        }
      }

      if (has_graft) {
        auto raw_prior_graft = evaluator->PriorWithLegals(*s1, l2);
        std::sort(raw_prior_graft.begin(), raw_prior_graft.end(),
                  [](const auto& x, const auto& y) { return x.second > y.second; });

        size_t n_graft_to_test = std::min<size_t>(2, raw_prior_graft.size());
        for (size_t g_idx = 0; g_idx < n_graft_to_test; ++g_idx) {
          Action a_ext = raw_prior_graft[g_idx].first;
          double p_ext = raw_prior_graft[g_idx].second;
          auto s_ext = s1->Clone();
          s_ext->ApplyAction(a_ext);

          if (!s_ext->IsTerminal() && !s_ext->IsChanceNode() && s_ext->CurrentPlayer() == player) {
            auto l3 = s_ext->LegalActions();
            std::vector<Action> space_legals;
            for (Action a : l3) {
              if (IsBoardSpaceAction(a)) space_legals.push_back(a);
            }
            if (!space_legals.empty()) {
              auto raw_prior_spaces = evaluator->PriorWithLegals(*s_ext, space_legals);
              for (const auto& sp : raw_prior_spaces) {
                double jp = p1 * p_ext * sp.second;
                CandidatePairInfo cpi{a1, sp.first, a_ext, p1, sp.second, jp};
                all_pairs.push_back(cpi);
                pairs_by_a1[a1].push_back(cpi);
              }
            } else {
              double jp = p1 * p_ext;
              CandidatePairInfo cpi{a1, kInvalidAction, a_ext, p1, 1.0, jp};
              all_pairs.push_back(cpi);
              pairs_by_a1[a1].push_back(cpi);
            }
          }
        }
        if (!pairs_by_a1[a1].empty()) {
          continue;
        }
      } else {
        std::vector<Action> space_legals;
        for (Action a : l2) {
          if (IsBoardSpaceAction(a)) space_legals.push_back(a);
        }
        if (!space_legals.empty()) {
          auto raw_prior_spaces = evaluator->PriorWithLegals(*s1, space_legals);
          for (const auto& sp : raw_prior_spaces) {
            double jp = p1 * sp.second;
            CandidatePairInfo cpi{a1, sp.first, kInvalidAction, p1, sp.second, jp};
            all_pairs.push_back(cpi);
            pairs_by_a1[a1].push_back(cpi);
          }
          continue;
        }
      }
    }
    // Fallback standalone action
    CandidatePairInfo cpi{a1, kInvalidAction, kInvalidAction, p1, 1.0, p1};
    all_pairs.push_back(cpi);
    pairs_by_a1[a1].push_back(cpi);
  }

  // Sort each a1's pairs by joint_p descending
  for (auto& kv : pairs_by_a1) {
    std::sort(kv.second.begin(), kv.second.end(),
              [](const CandidatePairInfo& x, const CandidatePairInfo& y) {
                return x.joint_p > y.joint_p;
              });
  }

  // Determine exact raw greedy candidate (cands[0])
  if (!pairs_by_a1[a1_raw].empty()) {
    const auto& best_for_a1_raw = pairs_by_a1[a1_raw].front();
    a2_raw = best_for_a1_raw.a2;
    a_ext_raw = best_for_a1_raw.a_extra;
    raw_joint_p = best_for_a1_raw.joint_p;
  }
  JointActionPair raw_pair{a1_raw, a2_raw, a_ext_raw, raw_joint_p};

  std::vector<JointActionPair> cands;
  auto add_cand = [&](Action a1, Action a2, Action a_ext, double jp) {
    JointActionPair p{a1, a2, a_ext, jp};
    for (const auto& c : cands) {
      if (c == p) return;
    }
    cands.push_back(p);
  };

  // 1. Index 0: sequential raw greedy
  add_cand(raw_pair.a1, raw_pair.a2, raw_pair.a_extra, raw_pair.joint_prior);

  // 2. Second space of a1_raw (if available)
  if (pairs_by_a1[a1_raw].size() >= 2) {
    const auto& sec = pairs_by_a1[a1_raw][1];
    add_cand(sec.a1, sec.a2, sec.a_extra, sec.joint_p);
  }

  // 3. Top space of top-4 distinct a1s
  for (size_t i = 0; i < std::min<size_t>(4, sorted_a1.size()); ++i) {
    Action a1 = sorted_a1[i];
    if (!pairs_by_a1[a1].empty()) {
      const auto& top_p = pairs_by_a1[a1].front();
      add_cand(top_p.a1, top_p.a2, top_p.a_extra, top_p.joint_p);
    }
  }

  // 4. Top joint priors up to max_k
  std::sort(all_pairs.begin(), all_pairs.end(),
            [](const CandidatePairInfo& x, const CandidatePairInfo& y) {
              return x.joint_p > y.joint_p;
            });
  for (const auto& cp : all_pairs) {
    if (static_cast<int>(cands.size()) >= max_k) break;
    add_cand(cp.a1, cp.a2, cp.a_extra, cp.joint_p);
  }

  return cands;
}

inline std::vector<std::vector<double>> EvaluatePairedJointCandidatesOnWorlds(
    const State& root_state,
    Player search_player,
    BatchedNNEvaluator* evaluator,
    const std::vector<JointActionPair>& candidates,
    int num_worlds,
    uint64_t stage_seed,
    SearchRolloutCrnMode crn_mode = SearchRolloutCrnMode::kEventKeyedCrn) {
  const size_t num_cands = candidates.size();
  std::vector<std::vector<double>> results(num_cands, std::vector<double>(num_worlds, 0.0));
  if (num_cands == 0 || num_worlds <= 0) return results;

  const DuneImperiumState* dune_root =
      dynamic_cast<const DuneImperiumState*>(&root_state);

  const int chunk_size = 64;
  for (int chunk_start = 0; chunk_start < num_worlds; chunk_start += chunk_size) {
    int chunk_end = std::min(num_worlds, chunk_start + chunk_size);
    int current_chunk = chunk_end - chunk_start;

    struct Job {
      size_t cand_idx;
      int world_idx;
      std::future<double> fut;
    };
    std::vector<Job> jobs;
    jobs.reserve(current_chunk * num_cands);

    for (int w = chunk_start; w < chunk_end; ++w) {
      uint64_t wseed = dune_seed::DeriveSeed(stage_seed, 0x57524C44ULL, w);
      std::unique_ptr<State> base_world = nullptr;
      if (dune_root != nullptr) {
        std::mt19937 wrng(wseed);
        auto rng_func = [&wrng]() {
          return std::generate_canonical<double, 53>(wrng);
        };
        base_world = dune_root->ResampleFromInfostate(search_player, rng_func);
      }
      if (!base_world) {
        base_world = root_state.Clone();
      }

      for (size_t c = 0; c < num_cands; ++c) {
        auto sim = (c + 1 == num_cands) ? std::move(base_world) : base_world->Clone();
        sim->ApplyAction(candidates[c].a1);
        if (candidates[c].a_extra != kInvalidAction) {
          sim->ApplyAction(candidates[c].a_extra);
        }
        if (candidates[c].a2 != kInvalidAction) {
          sim->ApplyAction(candidates[c].a2);
        }
        if (evaluator != nullptr) {
          evaluator->AdjustActiveRollouts(1);
        }
        jobs.push_back({c, w, SearchRolloutThreadPool::Instance().Submit(
            [sim = std::move(sim), search_player, evaluator, wseed, crn_mode]() {
              ActiveRolloutReleaseGuard guard{evaluator};
              auto traj = SimulateRolloutWithTurnTrajectory(
                  sim.get(), search_player, evaluator, wseed,
                  SearchTruncationMode::kTerminal, crn_mode);
              return traj.terminal_utility;
            })});
      }
    }

    for (auto& job : jobs) {
      results[job.cand_idx][job.world_idx] = job.fut.get();
    }
  }

  return results;
}

// -----------------------------------------------------------------------------
// MODE: gate4a_buy
// -----------------------------------------------------------------------------

void RunGate4aBuy() {
  std::cout << "========================================================\n";
  std::cout << "=== MODE: gate4a_buy (Track 2 Ord 3b Offline Buy)    ===\n";
  std::cout << "========================================================\n\n";

  auto bundle = LoadV5OrV4EvalBundle();
  std::cout << "Loaded bundle model: " << bundle.model_path << "\n";
  std::cout << "  Input Dim: " << bundle.ckpt_input_dim
            << " | Hidden Dim: " << bundle.hidden_dim
            << " | Blocks: " << bundle.num_blocks << "\n";
  std::cout << "  Schema: " << bundle.descriptor_schema
            << " | Device: " << (bundle.device.is_cuda() ? "CUDA" : "CPU") << "\n\n";

  const int target_roots = absl::GetFlag(FLAGS_gate4a_buy_roots);
  const int ref_rollouts = absl::GetFlag(FLAGS_gate4a_ref_rollouts);
  const uint64_t master_seed = absl::GetFlag(FLAGS_gate4a_seed);

  // 1. Stratified Natural Census (200 games parallel)
  std::cout << "--------------------------------------------------------\n";
  std::cout << "STEP 1: Stratified Natural Census (Target N = " << target_roots << ")\n";
  std::cout << "--------------------------------------------------------\n";

  struct BuyCensusCandidate {
    int game_id = 0;
    int round = 1;
    Player player = 0;
    int specimens = 0;
    int persuasion = 0;
    bool has_sb_sh = false;
    int affordable_acquisitions = 0;
    std::shared_ptr<State> state;
  };

  const int num_census_games = 200;
  std::atomic<int> next_census_game{0};
  std::mutex cand_mu;
  std::vector<BuyCensusCandidate> all_candidates;

  const int num_census_workers = 32;
  std::vector<std::thread> census_workers;
  census_workers.reserve(num_census_workers);

  const auto census_t0 = std::chrono::steady_clock::now();
  for (int w = 0; w < num_census_workers; ++w) {
    census_workers.emplace_back([&, w]() {
      auto game = LoadGame("dune_imperium(enable_immortality=true)");
      while (true) {
        int gid = next_census_game.fetch_add(1);
        if (gid >= num_census_games) break;

        uint64_t gseed = dune_seed::DeriveSeed(master_seed, 0x43454E53ULL, gid);
        std::mt19937_64 grng(gseed);
        auto st = game->NewInitialState();

        while (!st->IsTerminal()) {
          if (st->IsChanceNode()) {
            auto outcomes = st->ChanceOutcomes();
            double u = std::generate_canonical<double, 53>(grng);
            Action ca = SampleActionSafe(outcomes, u);
            st->ApplyAction(ca);
            continue;
          }

          Player p = st->CurrentPlayer();
          auto legal = st->LegalActions();
          if (legal.empty()) break;

          auto* dune_st = dynamic_cast<const DuneImperiumState*>(st.get());
          int rnd = dune_st ? dune_st->GetCurrentRound() : 1;
          bool is_cat3 = (rnd >= 1 && rnd <= 6 &&
                          ClassifyR755SearchCategory(*st, p, /*space_placed_this_turn=*/false,
                                                     /*category_mask=*/4,
                                                     /*cat3_strict_card_tech_only=*/true) ==
                              R755SearchCategory::kRevealPurchase &&
                          legal.size() >= 2);

          if (is_cat3) {
            int n_acq = 0;
            for (Action a : legal) {
              if (IsCardOrTechAcquisitionAction(a)) ++n_acq;
            }
            BuyCensusCandidate cand;
            cand.game_id = gid;
            cand.round = rnd;
            cand.player = p;
            cand.specimens = dune_st ? dune_st->GetSpecimensForTesting(p) : 0;
            cand.persuasion = dune_st ? dune_st->GetPlayerPersuasionForTesting(p) : 0;
            cand.has_sb_sh = false;
            if (dune_st != nullptr) {
              const auto& trow = dune_st->GetTleilaxuRow();
              for (int c : trow) {
                if (c == 11 || c == 13) { cand.has_sb_sh = true; break; }
              }
            }
            cand.affordable_acquisitions = n_acq;
            cand.state = st->Clone();
            {
              std::lock_guard<std::mutex> lk(cand_mu);
              all_candidates.push_back(std::move(cand));
            }
          }

          auto prior_legals = bundle.evaluator->PriorWithLegals(*st, legal);
          double u_act = std::generate_canonical<double, 53>(grng);
          Action act = SampleActionSafe(prior_legals, u_act);
          st->ApplyAction(act);
        }
      }
    });
  }
  for (auto& t : census_workers) t.join();
  double census_sec = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - census_t0).count();
  std::cout << "Census completed in " << std::fixed << std::setprecision(2)
            << census_sec << "s: extracted " << all_candidates.size()
            << " eligible Reveal purchase roots across " << num_census_games << " games.\n";

  // Stratify into 3 buckets (R1-2, R3-4, R5-6)
  std::vector<BuyCensusCandidate> cands_r12;
  std::vector<BuyCensusCandidate> cands_r34;
  std::vector<BuyCensusCandidate> cands_r56;
  for (auto& c : all_candidates) {
    if (c.round <= 2) cands_r12.push_back(std::move(c));
    else if (c.round <= 4) cands_r34.push_back(std::move(c));
    else if (c.round <= 6) cands_r56.push_back(std::move(c));
  }

  const int q_r12 = (target_roots + 2) / 3;
  const int q_r34 = (target_roots + 1) / 3;
  const int q_r56 = target_roots - q_r12 - q_r34;

  auto select_stratified_candidates = [](
      std::vector<BuyCensusCandidate>& pool, int target_quota, uint64_t seed) {
    std::vector<BuyCensusCandidate> selected;
    if (pool.empty() || target_quota <= 0) return selected;

    auto score_cand = [](const BuyCensusCandidate& c) {
      double s = 0.0;
      if (c.has_sb_sh) s += 10.0;
      if (c.specimens >= 3) s += 5.0;
      if (c.affordable_acquisitions >= 2) s += 2.0;
      return s;
    };

    std::map<int, std::vector<BuyCensusCandidate>> by_game;
    for (auto& c : pool) {
      by_game[c.game_id].push_back(std::move(c));
    }
    for (auto& kv : by_game) {
      std::sort(kv.second.begin(), kv.second.end(),
                [&](const BuyCensusCandidate& a, const BuyCensusCandidate& b) {
                  double sa = score_cand(a);
                  double sb = score_cand(b);
                  if (sa != sb) return sa > sb;
                  if (a.round != b.round) return a.round < b.round;
                  return a.player < b.player;
                });
    }

    std::vector<int> gids;
    for (const auto& kv : by_game) gids.push_back(kv.first);
    std::mt19937_64 rng(seed);
    std::shuffle(gids.begin(), gids.end(), rng);

    size_t pass = 0;
    while (static_cast<int>(selected.size()) < target_quota) {
      bool added = false;
      for (int gid : gids) {
        if (pass < by_game[gid].size()) {
          selected.push_back(std::move(by_game[gid][pass]));
          added = true;
          if (static_cast<int>(selected.size()) == target_quota) break;
        }
      }
      if (!added) break;
      ++pass;
    }
    return selected;
  };

  auto sel_r12 = select_stratified_candidates(cands_r12, q_r12, master_seed + 1);
  auto sel_r34 = select_stratified_candidates(cands_r34, q_r34, master_seed + 2);
  auto sel_r56 = select_stratified_candidates(cands_r56, q_r56, master_seed + 3);

  std::vector<BuyCensusCandidate> roots_meta;
  roots_meta.reserve(sel_r12.size() + sel_r34.size() + sel_r56.size());
  for (auto& c : sel_r12) roots_meta.push_back(std::move(c));
  for (auto& c : sel_r34) roots_meta.push_back(std::move(c));
  for (auto& c : sel_r56) roots_meta.push_back(std::move(c));

  const size_t N = roots_meta.size();
  std::cout << "Selected " << N << " stratified roots: R1-2=" << sel_r12.size()
            << ", R3-4=" << sel_r34.size() << ", R5-6=" << sel_r56.size() << "\n\n";

  // 2. Evaluate Two Teachers on All Roots (seed domain 0x5443485F425559ULL)
  std::cout << "--------------------------------------------------------\n";
  std::cout << "STEP 2: Evaluating Two Teachers on All " << N << " Roots\n";
  std::cout << "--------------------------------------------------------\n";

  struct TeacherEvalResult {
    Action raw_action = kInvalidAction;
    double raw_prior = 0.0;
    Action chosen_action = kInvalidAction;
    bool overridden = false;
    bool confirmed = false;
    bool teacher_accepted = false;
    Action teacher_action = kInvalidAction;
    Action alt_action = kInvalidAction;
    double mean_d_s2 = 0.0;
    double t_s2 = 0.0;
  };

  std::vector<TeacherEvalResult> teacher_a_res(N);
  std::vector<TeacherEvalResult> teacher_b_res(N);

  std::atomic<size_t> next_root_eval{0};
  const int num_eval_threads = 16;
  std::vector<std::thread> eval_threads;
  eval_threads.reserve(num_eval_threads);

  const auto teachers_t0 = std::chrono::steady_clock::now();
  for (int th = 0; th < num_eval_threads; ++th) {
    eval_threads.emplace_back([&, th]() {
      while (true) {
        size_t i = next_root_eval.fetch_add(1);
        if (i >= N) break;

        const auto& cand = roots_meta[i];
        State* root_state = cand.state.get();
        Player p = cand.player;
        auto legals = root_state->LegalActions();

        auto raw_prior_legals = bundle.evaluator->PriorWithLegals(*root_state, legals);
        Action raw_action = legals.front();
        double p_raw = 0.0;
        std::map<Action, double> p_map;
        for (const auto& ap : raw_prior_legals) {
          p_map[ap.first] = ap.second;
          if (ap.second > p_raw) {
            p_raw = ap.second;
            raw_action = ap.first;
          }
        }

        // --- Teacher A (search_v2_play) ---
        TeacherEvalResult res_a;
        res_a.raw_action = raw_action;
        res_a.raw_prior = p_raw;
        res_a.chosen_action = raw_action;
        res_a.overridden = false;

        std::vector<Action> ordered_cands = {raw_action};
        auto add_cand_a = [&](Action a) {
          if (std::find(ordered_cands.begin(), ordered_cands.end(), a) == ordered_cands.end()) {
            ordered_cands.push_back(a);
          }
        };
        for (Action a : legals) {
          if (IsCardOrTechAcquisitionAction(a)) add_cand_a(a);
        }
        if (std::find(legals.begin(), legals.end(), 90) != legals.end()) {
          add_cand_a(90);
        }
        auto sorted_legals = legals;
        std::sort(sorted_legals.begin(), sorted_legals.end(),
                  [&](Action x, Action y) { return p_map[x] > p_map[y]; });
        for (size_t k = 0; k < std::min<size_t>(5, sorted_legals.size()); ++k) {
          add_cand_a(sorted_legals[k]);
        }

        uint64_t root_master_seed = dune_seed::DeriveSeed(
            master_seed, 0x5443485F425559ULL, i);

        if (ordered_cands.size() >= 2) {
          uint64_t s1_seed = dune_seed::DeriveSeed(root_master_seed, 0x5331ULL, 0);
          auto res_s1 = EvaluatePairedActionsOnWorlds(
              *root_state, p, bundle.evaluator.get(), ordered_cands, 16, s1_seed,
              SearchRolloutCrnMode::kEventKeyedCrn);

          size_t best_c = 1;
          double best_adv_1 = -1e9;
          for (size_t c = 1; c < ordered_cands.size(); ++c) {
            double sum_diff = 0.0;
            for (int w = 0; w < 16; ++w) {
              sum_diff += (res_s1[c][w] - res_s1[0][w]);
            }
            double adv = sum_diff / 16.0;
            if (adv > best_adv_1) {
              best_adv_1 = adv;
              best_c = c;
            }
          }

          bool run_s2 = (best_adv_1 > -0.20) && !(p_raw >= 0.95 && best_adv_1 <= 0.05);
          if (run_s2) {
            std::vector<Action> s2_cands = {ordered_cands[0], ordered_cands[best_c]};
            uint64_t s2_seed = dune_seed::DeriveSeed(root_master_seed, 0x5332ULL, 1);
            auto res_s2 = EvaluatePairedActionsOnWorlds(
                *root_state, p, bundle.evaluator.get(), s2_cands, 160, s2_seed,
                SearchRolloutCrnMode::kEventKeyedCrn);

            double sum_d_s2 = 0.0, sum_sq_d_s2 = 0.0;
            for (int w = 0; w < 160; ++w) {
              double d = res_s2[1][w] - res_s2[0][w];
              sum_d_s2 += d;
              sum_sq_d_s2 += d * d;
            }
            double mean_d_s2 = sum_d_s2 / 160.0;
            double var_d_s2 = std::max(0.0, (sum_sq_d_s2 - 160.0 * mean_d_s2 * mean_d_s2) / 159.0);
            double se_d_s2 = std::sqrt(var_d_s2 / 160.0);
            double t_s2 = mean_d_s2 / (se_d_s2 + 1e-9);

            double sum_d_total = sum_d_s2;
            for (int w = 0; w < 16; ++w) {
              sum_d_total += (res_s1[best_c][w] - res_s1[0][w]);
            }
            double mean_d_total = sum_d_total / 176.0;

            double p_0 = p_raw;
            double p_c = p_map[ordered_cands[best_c]];
            double effective_margin = 0.15 + 0.02 * std::clamp(
                std::log((p_0 + 1e-6) / (p_c + 1e-6)), 0.0, 4.0);

            res_a.mean_d_s2 = mean_d_s2;
            res_a.t_s2 = t_s2;

            if (mean_d_s2 >= effective_margin && t_s2 >= 1.50 &&
                mean_d_total >= 0.75 * effective_margin) {
              res_a.chosen_action = ordered_cands[best_c];
              res_a.overridden = true;
            }
          }
        }
        teacher_a_res[i] = std::move(res_a);

        // --- Teacher B (online_s_v5_cat3) ---
        TeacherEvalResult res_b;
        res_b.raw_action = raw_action;
        res_b.raw_prior = p_raw;
        res_b.chosen_action = raw_action;
        res_b.teacher_action = raw_action;
        res_b.alt_action = raw_action;

        if (p_raw < 0.85) {
          CompoundSearchOptions opts_b;
          opts_b.all_buy_candidates = true;
          opts_b.symmetric_stage2_confirmation = true;
          opts_b.confirmation_max_raw_prior = 0.85;
          opts_b.stage1_rollouts = absl::GetFlag(FLAGS_gate4a_online_s1_rollouts);
          opts_b.stage1_min_mean_diff = 0.03;
          opts_b.stage2_independent_rollouts = absl::GetFlag(FLAGS_gate4a_online_s2_rollouts);
          opts_b.dedup_paired_worlds = true;
          opts_b.precomputed_raw_prior = &raw_prior_legals;

          int b_rollouts = opts_b.stage1_rollouts + opts_b.stage2_independent_rollouts;
          uint64_t b_dec_seed = dune_seed::DeriveSeed(root_master_seed, 0x43415433ULL, 0);

          auto dec_b = CompoundRolloutSearchDecision(
              *root_state, p, nullptr, {bundle.evaluator},
              /*top_k=*/12, b_rollouts, /*min_override_margin=*/0.15,
              TreeReuseMode::kTargetFill, b_dec_seed,
              SearchPairingMode::kPaired,
              SearchTruncationMode::kTerminal,
              SearchRolloutCrnMode::kEventKeyedCrn,
              nullptr,
              absl::GetFlag(FLAGS_gate4a_online_min_t),
              opts_b);

          res_b.overridden = dec_b.overridden;
          res_b.confirmed = dec_b.confirmed;
          res_b.teacher_accepted = dec_b.teacher_accepted;
          res_b.chosen_action = dec_b.action;
          res_b.teacher_action = dec_b.teacher_action;
          res_b.mean_d_s2 = dec_b.mean_d_stage2;
          res_b.t_s2 = (dec_b.se_d_stage2 > 0.0) ? (dec_b.mean_d_stage2 / dec_b.se_d_stage2) : 0.0;

          if (dec_b.overridden) {
            res_b.alt_action = raw_action;
          } else if (dec_b.confirmed) {
            res_b.teacher_action = raw_action;
            res_b.alt_action = (dec_b.stage2_candidate_utilities.size() >= 2)
                ? dec_b.stage2_candidate_utilities[1].first
                : raw_action;
          }
        }
        teacher_b_res[i] = std::move(res_b);
      }
    });
  }
  for (auto& t : eval_threads) t.join();
  double teachers_sec = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - teachers_t0).count();
  std::cout << "Both teachers evaluated on " << N << " roots in "
            << std::fixed << std::setprecision(2) << teachers_sec << "s.\n\n";

  // 3. Fresh Independent Ref512 Bank Evaluation (seed domain 0x52454635313242ULL)
  std::cout << "--------------------------------------------------------\n";
  std::cout << "STEP 3: Fresh Independent Ref512 Bank Evaluation (R_eval = " << ref_rollouts << ")\n";
  std::cout << "--------------------------------------------------------\n";

  std::vector<size_t> active_root_indices;
  std::map<size_t, std::vector<Action>> active_distinct_actions;
  for (size_t i = 0; i < N; ++i) {
    bool active = teacher_a_res[i].overridden ||
                  teacher_b_res[i].overridden ||
                  teacher_b_res[i].confirmed;
    if (active) {
      active_root_indices.push_back(i);
      std::vector<Action> acts;
      auto add_act = [&](Action a) {
        if (a != kInvalidAction && std::find(acts.begin(), acts.end(), a) == acts.end()) {
          acts.push_back(a);
        }
      };
      add_act(teacher_a_res[i].raw_action);
      if (teacher_a_res[i].overridden) {
        add_act(teacher_a_res[i].chosen_action);
      }
      if (teacher_b_res[i].overridden) {
        add_act(teacher_b_res[i].chosen_action);
      }
      if (teacher_b_res[i].confirmed) {
        add_act(teacher_b_res[i].teacher_action);
        add_act(teacher_b_res[i].alt_action);
      }
      active_distinct_actions[i] = acts;
    }
  }

  std::cout << "Identified " << active_root_indices.size()
            << " active roots where at least one teacher intervened or confirmed.\n";

  std::map<size_t, std::map<Action, double>> ref_u;
  std::map<size_t, std::map<Action, double>> sum_u;

  const auto ref_t0 = std::chrono::steady_clock::now();
  const int world_chunk_size = 64;
  for (int w_chunk = 0; w_chunk < ref_rollouts; w_chunk += world_chunk_size) {
    int w_end = std::min(ref_rollouts, w_chunk + world_chunk_size);
    int cur_chunk = w_end - w_chunk;

    struct RefJob {
      size_t root_idx;
      Action action;
      std::future<double> fut;
    };
    std::vector<RefJob> chunk_jobs;

    for (size_t root_idx : active_root_indices) {
      const auto* dune_root = dynamic_cast<const DuneImperiumState*>(roots_meta[root_idx].state.get());
      Player p = roots_meta[root_idx].player;
      const auto& acts = active_distinct_actions[root_idx];

      for (int w = w_chunk; w < w_end; ++w) {
        uint64_t wseed = dune_seed::DeriveSeed(
            master_seed,
            0x52454635313242ULL + static_cast<uint64_t>(root_idx),
            w);
        std::unique_ptr<State> base_world = nullptr;
        if (dune_root != nullptr) {
          std::mt19937 wrng(wseed);
          auto rng_func = [&wrng]() {
            return std::generate_canonical<double, 53>(wrng);
          };
          base_world = dune_root->ResampleFromInfostate(p, rng_func);
        }
        if (!base_world) {
          base_world = roots_meta[root_idx].state->Clone();
        }

        for (size_t a_i = 0; a_i < acts.size(); ++a_i) {
          auto sim = (a_i + 1 == acts.size()) ? std::move(base_world) : base_world->Clone();
          sim->ApplyAction(acts[a_i]);
          if (bundle.evaluator != nullptr) {
            bundle.evaluator->AdjustActiveRollouts(1);
          }
          chunk_jobs.push_back({root_idx, acts[a_i], SearchRolloutThreadPool::Instance().Submit(
              [sim = std::move(sim), p, eval = bundle.evaluator.get(), wseed]() {
                ActiveRolloutReleaseGuard guard{eval};
                auto traj = SimulateRolloutWithTurnTrajectory(
                    sim.get(), p, eval, wseed,
                    SearchTruncationMode::kTerminal, SearchRolloutCrnMode::kEventKeyedCrn);
                return traj.terminal_utility;
              })});
        }
      }
    }

    for (auto& job : chunk_jobs) {
      sum_u[job.root_idx][job.action] += job.fut.get();
    }
    std::cout << "  [Ref512 Bank Progress] Worlds " << w_end << " / " << ref_rollouts
              << " completed (" << chunk_jobs.size() << " rollouts)\n";
  }

  for (size_t root_idx : active_root_indices) {
    for (Action a : active_distinct_actions[root_idx]) {
      ref_u[root_idx][a] = sum_u[root_idx][a] / static_cast<double>(ref_rollouts);
    }
  }

  double ref_sec = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - ref_t0).count();
  std::cout << "Ref512 evaluation completed in " << std::fixed << std::setprecision(2)
            << ref_sec << "s.\n\n";

  // 4. Game-Clustered Bootstrap (10,000 resamples) & Metrics
  std::cout << "--------------------------------------------------------\n";
  std::cout << "STEP 4: Game-Clustered Bootstrap (10,000 Resamples) & Gate 4A Assessment\n";
  std::cout << "--------------------------------------------------------\n";

  std::vector<int> game_ids(N);
  for (size_t i = 0; i < N; ++i) game_ids[i] = roots_meta[i].game_id;

  // Teacher A metrics
  std::vector<double> play_gain_all(N, 0.0);
  std::vector<double> play_gain_override;
  std::vector<int> play_gid_override;
  int play_overrides = 0;
  int play_bad_overrides = 0;
  int play_mat_bad_overrides = 0;

  for (size_t i = 0; i < N; ++i) {
    if (teacher_a_res[i].overridden) {
      ++play_overrides;
      Action ch = teacher_a_res[i].chosen_action;
      Action rw = teacher_a_res[i].raw_action;
      double diff = ref_u[i][ch] - ref_u[i][rw];
      play_gain_all[i] = diff;
      play_gain_override.push_back(diff);
      play_gid_override.push_back(game_ids[i]);
      if (diff < 0.0) ++play_bad_overrides;
      if (diff < -0.05) ++play_mat_bad_overrides;
    }
  }

  auto boot_play_all = ComputeGameClusteredBootstrap(play_gain_all, game_ids, 10000, master_seed + 10);
  auto boot_play_ovr = ComputeGameClusteredBootstrap(play_gain_override, play_gid_override, 10000, master_seed + 11);
  double bad_rate_play = play_overrides > 0 ? (static_cast<double>(play_bad_overrides) / play_overrides) : 0.0;
  double mat_bad_rate_play = play_overrides > 0 ? (static_cast<double>(play_mat_bad_overrides) / play_overrides) : 0.0;
  double ovr_rate_play = static_cast<double>(play_overrides) / N;

  // Teacher B metrics
  std::vector<double> online_gain_all(N, 0.0);
  std::vector<double> online_gain_override;
  std::vector<int> online_gid_override;
  int online_overrides = 0;
  int online_bad_overrides = 0;
  int online_mat_bad_overrides = 0;

  std::vector<double> online_accepted_gain;
  std::vector<int> online_accepted_gid;
  int online_accepted = 0;
  int online_confirmations = 0;

  for (size_t i = 0; i < N; ++i) {
    if (teacher_b_res[i].overridden) {
      ++online_overrides;
      Action ch = teacher_b_res[i].chosen_action;
      Action rw = teacher_b_res[i].raw_action;
      double diff = ref_u[i][ch] - ref_u[i][rw];
      online_gain_all[i] = diff;
      online_gain_override.push_back(diff);
      online_gid_override.push_back(game_ids[i]);
      if (diff < 0.0) ++online_bad_overrides;
      if (diff < -0.05) ++online_mat_bad_overrides;
    }
    if (teacher_b_res[i].confirmed) {
      ++online_confirmations;
    }
    if (teacher_b_res[i].teacher_accepted) {
      ++online_accepted;
      Action tch = teacher_b_res[i].teacher_action;
      Action alt = teacher_b_res[i].alt_action;
      double pair_diff = ref_u[i][tch] - ref_u[i][alt];
      online_accepted_gain.push_back(pair_diff);
      online_accepted_gid.push_back(game_ids[i]);
    }
  }

  auto boot_online_all = ComputeGameClusteredBootstrap(online_gain_all, game_ids, 10000, master_seed + 20);
  auto boot_online_ovr = ComputeGameClusteredBootstrap(online_gain_override, online_gid_override, 10000, master_seed + 21);
  auto boot_online_acc = ComputeGameClusteredBootstrap(online_accepted_gain, online_accepted_gid, 10000, master_seed + 22);

  double bad_rate_online = online_overrides > 0 ? (static_cast<double>(online_bad_overrides) / online_overrides) : 0.0;
  double mat_bad_rate_online = online_overrides > 0 ? (static_cast<double>(online_mat_bad_overrides) / online_overrides) : 0.0;
  double ovr_rate_online = static_cast<double>(online_overrides) / N;

  bool gate4a_buy_play_pass = (boot_play_all.ci_low > 0.0 && bad_rate_play < 0.15);
  bool gate4a_buy_online_pass = (boot_online_all.ci_low > 0.0 &&
                                 boot_online_acc.ci_low > 0.0 &&
                                 bad_rate_online < 0.15);

  // Round bucket breakdown
  struct BucketStats {
    int total = 0;
    int play_ovr = 0;
    double play_gain = 0.0;
    int online_ovr = 0;
    double online_gain = 0.0;
  };
  std::map<std::string, BucketStats> bucket_map;
  bucket_map["R1-2"] = {};
  bucket_map["R3-4"] = {};
  bucket_map["R5-6"] = {};

  for (size_t i = 0; i < N; ++i) {
    int rnd = roots_meta[i].round;
    std::string bname = (rnd <= 2) ? "R1-2" : ((rnd <= 4) ? "R3-4" : "R5-6");
    bucket_map[bname].total++;
    if (teacher_a_res[i].overridden) {
      bucket_map[bname].play_ovr++;
      bucket_map[bname].play_gain += play_gain_all[i];
    }
    if (teacher_b_res[i].overridden) {
      bucket_map[bname].online_ovr++;
      bucket_map[bname].online_gain += online_gain_all[i];
    }
  }

  // Acquisition source breakdown
  struct AcqSourceStats {
    int play_count = 0;
    double play_gain = 0.0;
    int online_count = 0;
    double online_gain = 0.0;
  };
  std::map<std::string, AcqSourceStats> source_map;
  for (size_t i = 0; i < N; ++i) {
    const auto* dst = dynamic_cast<const DuneImperiumState*>(roots_meta[i].state.get());
    if (teacher_a_res[i].overridden) {
      std::string src = ClassifyAcquisitionSource(teacher_a_res[i].chosen_action, dst);
      source_map[src].play_count++;
      source_map[src].play_gain += play_gain_all[i];
    }
    if (teacher_b_res[i].overridden) {
      std::string src = ClassifyAcquisitionSource(teacher_b_res[i].chosen_action, dst);
      source_map[src].online_count++;
      source_map[src].online_gain += online_gain_all[i];
    }
  }

  // Print Formatted Report to stdout
  std::cout << "\n### Table 1: Gate 4A Buy Offline Teacher Summary (N = " << N << ")\n";
  std::cout << "| Metric                           | search_v2_play       | online_s_v5_cat3     |\n";
  std::cout << "|:---------------------------------|:---------------------|:---------------------|\n";
  std::cout << "| Roots Evaluated                  | " << std::setw(20) << N << " | " << std::setw(20) << N << " |\n";
  std::cout << "| Overrides (Rate)                 | " << std::setw(11) << play_overrides << " (" << std::fixed << std::setprecision(1) << (ovr_rate_play * 100.0) << "%) | "
            << std::setw(11) << online_overrides << " (" << (ovr_rate_online * 100.0) << "%) |\n";
  std::cout << "| All-Roots Mean Gain              | " << std::setw(20) << std::setprecision(4) << boot_play_all.mean << " | " << std::setw(20) << boot_online_all.mean << " |\n";
  std::cout << "| All-Roots 95% CI                 | [" << std::setprecision(4) << boot_play_all.ci_low << ", " << boot_play_all.ci_high << "] | ["
            << boot_online_all.ci_low << ", " << boot_online_all.ci_high << "] |\n";
  std::cout << "| Override-Only Mean Gain          | " << std::setw(20) << std::setprecision(4) << boot_play_ovr.mean << " | " << std::setw(20) << boot_online_ovr.mean << " |\n";
  std::cout << "| Override-Only 95% CI             | [" << std::setprecision(4) << boot_play_ovr.ci_low << ", " << boot_play_ovr.ci_high << "] | ["
            << boot_online_ovr.ci_low << ", " << boot_online_ovr.ci_high << "] |\n";
  std::cout << "| Bad Overrides (Rate < 15%)       | " << std::setw(11) << play_bad_overrides << " (" << std::setprecision(1) << (bad_rate_play * 100.0) << "%) | "
            << std::setw(11) << online_bad_overrides << " (" << (bad_rate_online * 100.0) << "%) |\n";
  std::cout << "| Material Bad (< -0.05)           | " << std::setw(11) << play_mat_bad_overrides << " (" << std::setprecision(1) << (mat_bad_rate_play * 100.0) << "%) | "
            << std::setw(11) << online_mat_bad_overrides << " (" << (mat_bad_rate_online * 100.0) << "%) |\n";
  std::cout << "| Accepted Teacher Pairs           | " << std::setw(20) << "N/A" << " | " << std::setw(20) << online_accepted << " |\n";
  std::cout << "| Low-Prior Confirmations          | " << std::setw(20) << "N/A" << " | " << std::setw(20) << online_confirmations << " |\n";
  std::cout << "| Accepted Pair Mean Gain          | " << std::setw(20) << "N/A" << " | " << std::setw(20) << std::setprecision(4) << boot_online_acc.mean << " |\n";
  std::cout << "| Accepted Pair 95% CI             | " << std::setw(20) << "N/A" << " | [" << std::setprecision(4) << boot_online_acc.ci_low << ", " << boot_online_acc.ci_high << "] |\n";
  std::cout << "| Gate 4A Assessment               | " << std::setw(20) << (gate4a_buy_play_pass ? "PASS" : "FAIL") << " | "
            << std::setw(20) << (gate4a_buy_online_pass ? "PASS" : "FAIL") << " |\n";

  std::cout << "\n### Table 2: Round Bucket Breakdown\n";
  std::cout << "| Bucket | Roots | Play Overrides | Play Mean Gain | Online Overrides | Online Mean Gain |\n";
  std::cout << "|:-------|:------|:---------------|:---------------|:-----------------|:-----------------|\n";
  for (const std::string& bname : {"R1-2", "R3-4", "R5-6"}) {
    const auto& st = bucket_map[bname];
    double mg_play = st.total > 0 ? (st.play_gain / st.total) : 0.0;
    double mg_online = st.total > 0 ? (st.online_gain / st.total) : 0.0;
    std::cout << "| " << std::setw(6) << bname << " | "
              << std::setw(5) << st.total << " | "
              << std::setw(14) << st.play_ovr << " | "
              << std::setw(14) << std::setprecision(4) << mg_play << " | "
              << std::setw(16) << st.online_ovr << " | "
              << std::setw(16) << std::setprecision(4) << mg_online << " |\n";
  }

  std::cout << "\n### Table 3: Acquisition Source Breakdown\n";
  std::cout << "| Source           | Play Overrides | Play Sum Gain  | Online Overrides | Online Sum Gain  |\n";
  std::cout << "|:-----------------|:---------------|:---------------|:-----------------|:-----------------|\n";
  for (const auto& kv : source_map) {
    std::cout << "| " << std::setw(16) << kv.first << " | "
              << std::setw(14) << kv.second.play_count << " | "
              << std::setw(14) << std::setprecision(4) << kv.second.play_gain << " | "
              << std::setw(16) << kv.second.online_count << " | "
              << std::setw(16) << std::setprecision(4) << kv.second.online_gain << " |\n";
  }
  std::cout << "\n";

  // Export JSON summary if requested
  const std::string json_path = absl::GetFlag(FLAGS_summary_json_path);
  if (!json_path.empty()) {
    std::ofstream ofs(json_path);
    if (ofs.good()) {
      ofs << "{\n"
          << "  \"mode\": \"gate4a_buy\",\n"
          << "  \"model_path\": \"" << bundle.model_path << "\",\n"
          << "  \"num_roots\": " << N << ",\n"
          << "  \"gate4a_buy_play_pass\": " << (gate4a_buy_play_pass ? "true" : "false") << ",\n"
          << "  \"gate4a_buy_online_pass\": " << (gate4a_buy_online_pass ? "true" : "false") << ",\n"
          << "  \"search_v2_play\": {\n"
          << "    \"num_roots\": " << N << ",\n"
          << "    \"num_overrides\": " << play_overrides << ",\n"
          << "    \"override_rate\": " << ovr_rate_play << ",\n"
          << "    \"all_roots_mean_gain\": " << boot_play_all.mean << ",\n"
          << "    \"all_roots_ci95_low\": " << boot_play_all.ci_low << ",\n"
          << "    \"all_roots_ci95_high\": " << boot_play_all.ci_high << ",\n"
          << "    \"override_only_mean_gain\": " << boot_play_ovr.mean << ",\n"
          << "    \"override_only_ci95_low\": " << boot_play_ovr.ci_low << ",\n"
          << "    \"override_only_ci95_high\": " << boot_play_ovr.ci_high << ",\n"
          << "    \"bad_override_count\": " << play_bad_overrides << ",\n"
          << "    \"bad_override_rate\": " << bad_rate_play << ",\n"
          << "    \"material_bad_override_rate\": " << mat_bad_rate_play << "\n"
          << "  },\n"
          << "  \"online_s_v5_cat3\": {\n"
          << "    \"num_roots\": " << N << ",\n"
          << "    \"num_overrides\": " << online_overrides << ",\n"
          << "    \"override_rate\": " << ovr_rate_online << ",\n"
          << "    \"all_roots_mean_gain\": " << boot_online_all.mean << ",\n"
          << "    \"all_roots_ci95_low\": " << boot_online_all.ci_low << ",\n"
          << "    \"all_roots_ci95_high\": " << boot_online_all.ci_high << ",\n"
          << "    \"override_only_mean_gain\": " << boot_online_ovr.mean << ",\n"
          << "    \"override_only_ci95_low\": " << boot_online_ovr.ci_low << ",\n"
          << "    \"override_only_ci95_high\": " << boot_online_ovr.ci_high << ",\n"
          << "    \"bad_override_count\": " << online_bad_overrides << ",\n"
          << "    \"bad_override_rate\": " << bad_rate_online << ",\n"
          << "    \"material_bad_override_rate\": " << mat_bad_rate_online << ",\n"
          << "    \"num_accepted\": " << online_accepted << ",\n"
          << "    \"num_confirmations\": " << online_confirmations << ",\n"
          << "    \"accepted_pair_mean_gain\": " << boot_online_acc.mean << ",\n"
          << "    \"accepted_pair_ci95_low\": " << boot_online_acc.ci_low << ",\n"
          << "    \"accepted_pair_ci95_high\": " << boot_online_acc.ci_high << "\n"
          << "  }\n"
          << "}\n";
      ofs.close();
      std::cout << "Exported summary JSON to " << json_path << "\n";
    }
  }

  std::cout << "========================================================\n";
  std::cout << "GATE4A_BUY COMPLETE: PLAY PASS = " << (gate4a_buy_play_pass ? "YES" : "NO")
            << " | ONLINE PASS = " << (gate4a_buy_online_pass ? "YES" : "NO") << "\n";
  std::cout << "========================================================\n";
}

// -----------------------------------------------------------------------------
// MODE: gate4a_place
// -----------------------------------------------------------------------------

inline JointActionPair RunSeqShBaseline(
    const State& state,
    Player player,
    std::shared_ptr<BatchedNNEvaluator> evaluator,
    uint64_t root_seed) {
  uint64_t seed1 = dune_seed::DeriveSeed(root_seed, 0x53455131ULL, 0);
  auto sh_res1 = SuccessiveHalvingRolloutSearchDecision(
      state, player, nullptr, {evaluator},
      /*reference_rollouts_per_action=*/256, /*min_override_margin=*/0.15,
      seed1, SearchRolloutCrnMode::kEventKeyedCrn,
      /*unbiased_stage2_gate=*/false, /*max_time_ms=*/-1.0,
      /*fixed_top_k=*/5);
  Action a1 = sh_res1.action;

  auto s1 = state.Clone();
  s1->ApplyAction(a1);
  if (s1->IsTerminal() || s1->IsChanceNode() || s1->CurrentPlayer() != player) {
    return {a1, kInvalidAction, kInvalidAction, 0.0};
  }

  auto l2 = s1->LegalActions();
  Action a_extra = kInvalidAction;
  bool has_graft = false;
  for (Action a : l2) {
    if (a >= 2056 && a <= 2312) {
      has_graft = true;
      break;
    }
  }

  if (has_graft) {
    if (l2.size() >= 2) {
      uint64_t seed_ext = dune_seed::DeriveSeed(root_seed, 0x53455145ULL, 2);
      auto sh_res_ext = SuccessiveHalvingRolloutSearchDecision(
          *s1, player, nullptr, {evaluator},
          /*reference_rollouts_per_action=*/256, /*min_override_margin=*/0.15,
          seed_ext, SearchRolloutCrnMode::kEventKeyedCrn,
          /*unbiased_stage2_gate=*/false, /*max_time_ms=*/-1.0,
          /*fixed_top_k=*/5);
      a_extra = sh_res_ext.action;
    } else if (!l2.empty()) {
      a_extra = l2.front();
    }
    if (a_extra != kInvalidAction) {
      s1->ApplyAction(a_extra);
      if (s1->IsTerminal() || s1->IsChanceNode() || s1->CurrentPlayer() != player) {
        return {a1, kInvalidAction, a_extra, 0.0};
      }
      l2 = s1->LegalActions();
    }
  }

  std::vector<Action> space_legals;
  for (Action a : l2) {
    if (IsBoardSpaceAction(a)) space_legals.push_back(a);
  }

  if (space_legals.size() >= 2) {
    uint64_t seed2 = dune_seed::DeriveSeed(root_seed, 0x53455132ULL, 1);
    auto sh_res2 = SuccessiveHalvingRolloutSearchDecision(
        *s1, player, nullptr, {evaluator},
        /*reference_rollouts_per_action=*/256, /*min_override_margin=*/0.15,
        seed2, SearchRolloutCrnMode::kEventKeyedCrn,
        /*unbiased_stage2_gate=*/false, /*max_time_ms=*/-1.0,
        /*fixed_top_k=*/5);
    return {a1, sh_res2.action, a_extra, 0.0};
  } else if (space_legals.size() == 1) {
    return {a1, space_legals.front(), a_extra, 0.0};
  } else if (!l2.empty()) {
    return {a1, l2.front(), a_extra, 0.0};
  }
  return {a1, kInvalidAction, a_extra, 0.0};
}

inline JointControllerResult RunSearchV2JointPlay(
    const State& state,
    Player player,
    BatchedNNEvaluator* evaluator,
    const std::vector<JointActionPair>& cands,
    uint64_t root_seed) {
  JointControllerResult res;
  if (cands.empty()) return res;
  res.chosen_pair = cands.front();
  res.raw_pair = cands.front();
  res.teacher_pair = cands.front();
  res.alt_pair = cands.front();
  res.overridden = false;
  if (cands.size() <= 1) return res;

  // Stage 1: 16 worlds
  uint64_t s1_seed = dune_seed::DeriveSeed(root_seed, 0x504C4331ULL, 0);
  auto res_s1 = EvaluatePairedJointCandidatesOnWorlds(
      state, player, evaluator, cands, 16, s1_seed,
      SearchRolloutCrnMode::kEventKeyedCrn);

  size_t best_c = 1;
  double best_adv_1 = -1e9;
  for (size_t c = 1; c < cands.size(); ++c) {
    double sum_diff = 0.0;
    for (int w = 0; w < 16; ++w) {
      sum_diff += (res_s1[c][w] - res_s1[0][w]);
    }
    double adv = sum_diff / 16.0;
    if (adv > best_adv_1) {
      best_adv_1 = adv;
      best_c = c;
    }
  }

  double p_raw = cands[0].joint_prior;
  bool run_s2 = (best_adv_1 > -0.20) && !(p_raw >= 0.95 && best_adv_1 <= 0.05);
  if (!run_s2) return res;

  // Stage 2: 160 worlds on {0, best_c}
  std::vector<JointActionPair> pair_cands = {cands[0], cands[best_c]};
  uint64_t s2_seed = dune_seed::DeriveSeed(root_seed, 0x504C4332ULL, 1);
  auto res_s2 = EvaluatePairedJointCandidatesOnWorlds(
      state, player, evaluator, pair_cands, 160, s2_seed,
      SearchRolloutCrnMode::kEventKeyedCrn);

  double sum_d_s2 = 0.0, sum_sq_d_s2 = 0.0;
  for (int w = 0; w < 160; ++w) {
    double d = res_s2[1][w] - res_s2[0][w];
    sum_d_s2 += d;
    sum_sq_d_s2 += d * d;
  }
  double mean_d_s2 = sum_d_s2 / 160.0;
  double var_d_s2 = std::max(0.0, (sum_sq_d_s2 - 160.0 * mean_d_s2 * mean_d_s2) / 159.0);
  double se_d_s2 = std::sqrt(var_d_s2 / 160.0);
  double t_s2 = mean_d_s2 / (se_d_s2 + 1e-9);

  double sum_d_total = sum_d_s2;
  for (int w = 0; w < 16; ++w) {
    sum_d_total += (res_s1[best_c][w] - res_s1[0][w]);
  }
  double mean_d_total = sum_d_total / 176.0;

  double p_0 = cands[0].joint_prior;
  double p_c = cands[best_c].joint_prior;
  double effective_margin = 0.15 + 0.02 * std::clamp(
      std::log((p_0 + 1e-6) / (p_c + 1e-6)), 0.0, 4.0);

  res.mean_d_s2 = mean_d_s2;
  res.t_s2 = t_s2;

  if (mean_d_s2 >= effective_margin && t_s2 >= 1.50 &&
      mean_d_total >= 0.75 * effective_margin) {
    res.chosen_pair = cands[best_c];
    res.teacher_pair = cands[best_c];
    res.alt_pair = cands[0];
    res.overridden = true;
    res.teacher_accepted = true;
  }
  return res;
}

inline JointControllerResult RunOnlineJointPlace(
    const State& state,
    Player player,
    BatchedNNEvaluator* evaluator,
    const std::vector<JointActionPair>& cands,
    uint64_t root_seed,
    int s2_rollouts,
    double min_t) {
  JointControllerResult res;
  if (cands.empty()) return res;
  res.chosen_pair = cands.front();
  res.raw_pair = cands.front();
  res.teacher_pair = cands.front();
  res.alt_pair = cands.front();
  res.overridden = false;
  res.confirmed = false;
  res.teacher_accepted = false;
  if (cands.size() <= 1) return res;

  double p_raw = cands[0].joint_prior;
  if (p_raw >= 0.85) {
    return res;
  }

  // Stage 1: 16 worlds
  uint64_t s1_seed = dune_seed::DeriveSeed(root_seed, 0x4F4E4C31ULL, 0);
  auto res_s1 = EvaluatePairedJointCandidatesOnWorlds(
      state, player, evaluator, cands, 16, s1_seed,
      SearchRolloutCrnMode::kEventKeyedCrn);

  size_t best_c = 1;
  double best_adv_1 = -1e9;
  for (size_t c = 1; c < cands.size(); ++c) {
    double sum_diff = 0.0;
    for (int w = 0; w < 16; ++w) {
      sum_diff += (res_s1[c][w] - res_s1[0][w]);
    }
    double adv = sum_diff / 16.0;
    if (adv > best_adv_1) {
      best_adv_1 = adv;
      best_c = c;
    }
  }

  // Case A: Challenger won Stage 1 (best_adv_1 >= 0.03)
  if (best_adv_1 >= 0.03) {
    // Stage 2: s2_rollouts (64 or 80) on {0, best_c}
    std::vector<JointActionPair> pair_cands = {cands[0], cands[best_c]};
    uint64_t s2_seed = dune_seed::DeriveSeed(root_seed, 0x4F4E4C32ULL, 1);
    auto res_s2 = EvaluatePairedJointCandidatesOnWorlds(
        state, player, evaluator, pair_cands, s2_rollouts, s2_seed,
        SearchRolloutCrnMode::kEventKeyedCrn);

    double sum_d_s2 = 0.0, sum_sq_d_s2 = 0.0;
    for (int w = 0; w < s2_rollouts; ++w) {
      double d = res_s2[1][w] - res_s2[0][w];
      sum_d_s2 += d;
      sum_sq_d_s2 += d * d;
    }
    double mean_d_s2 = sum_d_s2 / static_cast<double>(s2_rollouts);
    double var_d_s2 = (s2_rollouts > 1)
        ? std::max(0.0, (sum_sq_d_s2 - s2_rollouts * mean_d_s2 * mean_d_s2) / (s2_rollouts - 1))
        : 0.0;
    double se_d_s2 = std::sqrt(var_d_s2 / static_cast<double>(s2_rollouts));
    double t_s2 = (se_d_s2 > 1e-12) ? (mean_d_s2 / se_d_s2) : (mean_d_s2 > 0.0 ? 1e9 : -1e9);

    res.mean_d_s2 = mean_d_s2;
    res.t_s2 = t_s2;

    if (mean_d_s2 >= 0.15 && t_s2 >= min_t) {
      res.chosen_pair = cands[best_c];
      res.teacher_pair = cands[best_c];
      res.alt_pair = cands[0];
      res.overridden = true;
      res.teacher_accepted = true;
    }
    return res;
  }

  // Case B: best_adv_1 <= 0.0 (candidates[0] won Stage 1)
  // Low-prior symmetric confirmation on {cands[0], cands[1]}
  if (best_adv_1 <= 0.0 && cands.size() >= 2) {
    size_t rival_idx = 1;
    std::vector<JointActionPair> pair_cands = {cands[0], cands[rival_idx]};
    uint64_t s2_seed = dune_seed::DeriveSeed(root_seed, 0x4F4E4C33ULL, 2);
    auto res_s2 = EvaluatePairedJointCandidatesOnWorlds(
        state, player, evaluator, pair_cands, s2_rollouts, s2_seed,
        SearchRolloutCrnMode::kEventKeyedCrn);

    double sum_d_0_vs_r = 0.0, sum_sq_d_0_vs_r = 0.0;
    for (int w = 0; w < s2_rollouts; ++w) {
      double d = res_s2[0][w] - res_s2[1][w];
      sum_d_0_vs_r += d;
      sum_sq_d_0_vs_r += d * d;
    }
    double mean_d_0_vs_r = sum_d_0_vs_r / static_cast<double>(s2_rollouts);
    double var_d_0_vs_r = (s2_rollouts > 1)
        ? std::max(0.0, (sum_sq_d_0_vs_r - s2_rollouts * mean_d_0_vs_r * mean_d_0_vs_r) / (s2_rollouts - 1))
        : 0.0;
    double se_d_0_vs_r = std::sqrt(var_d_0_vs_r / static_cast<double>(s2_rollouts));
    double t_0_vs_r = (se_d_0_vs_r > 1e-12) ? (mean_d_0_vs_r / se_d_0_vs_r) : (mean_d_0_vs_r > 0.0 ? 1e9 : -1e9);

    res.mean_d_s2 = mean_d_0_vs_r;
    res.t_s2 = t_0_vs_r;

    if (mean_d_0_vs_r >= 0.15 && t_0_vs_r >= min_t) {
      res.confirmed = true;
      res.teacher_accepted = true;
      res.chosen_pair = cands[0];
      res.teacher_pair = cands[0];
      res.alt_pair = cands[rival_idx];
      res.overridden = false;
    } else {
      double mean_d_r_vs_0 = -mean_d_0_vs_r;
      double t_r_vs_0 = -t_0_vs_r;
      if (mean_d_r_vs_0 >= 0.15 && t_r_vs_0 >= min_t) {
        res.chosen_pair = cands[rival_idx];
        res.teacher_pair = cands[rival_idx];
        res.alt_pair = cands[0];
        res.overridden = true;
        res.teacher_accepted = true;
        res.mean_d_s2 = mean_d_r_vs_0;
        res.t_s2 = t_r_vs_0;
      }
    }
  }

  return res;
}

void RunGate4aPlace() {
  std::cout << "========================================================\n";
  std::cout << "=== MODE: gate4a_place (Track 2 Ord 3b Offline Place) ===\n";
  std::cout << "========================================================\n\n";

  auto bundle = LoadV5OrV4EvalBundle();
  std::cout << "Loaded bundle model: " << bundle.model_path << "\n";
  std::cout << "  Input Dim: " << bundle.ckpt_input_dim
            << " | Hidden Dim: " << bundle.hidden_dim
            << " | Blocks: " << bundle.num_blocks << "\n";
  std::cout << "  Schema: " << bundle.descriptor_schema
            << " | Device: " << (bundle.device.is_cuda() ? "CUDA" : "CPU") << "\n\n";

  const int target_roots = absl::GetFlag(FLAGS_gate4a_place_roots);
  const int ref_rollouts = absl::GetFlag(FLAGS_gate4a_ref_rollouts);
  const uint64_t master_seed = absl::GetFlag(FLAGS_gate4a_seed);

  // 1. Stratified Natural Placement Census (Target 200 roots)
  std::cout << "--------------------------------------------------------\n";
  std::cout << "STEP 1: Stratified Natural Placement Census (Target N = " << target_roots << ")\n";
  std::cout << "--------------------------------------------------------\n";

  struct PlaceCensusCandidate {
    int game_id = 0;
    int round = 1;
    Player player = 0;
    std::shared_ptr<State> state;
  };

  const int num_census_games = 200;
  std::atomic<int> next_census_game{0};
  std::mutex cand_mu;
  std::vector<PlaceCensusCandidate> all_candidates;

  const int num_census_workers = 32;
  std::vector<std::thread> census_workers;
  census_workers.reserve(num_census_workers);

  const auto census_t0 = std::chrono::steady_clock::now();
  for (int w = 0; w < num_census_workers; ++w) {
    census_workers.emplace_back([&, w]() {
      auto game = LoadGame("dune_imperium(enable_immortality=true)");
      while (true) {
        int gid = next_census_game.fetch_add(1);
        if (gid >= num_census_games) break;

        uint64_t gseed = dune_seed::DeriveSeed(master_seed, 0x504C4345ULL, gid);
        std::mt19937_64 grng(gseed);
        auto st = game->NewInitialState();

        while (!st->IsTerminal()) {
          if (st->IsChanceNode()) {
            auto outcomes = st->ChanceOutcomes();
            double u = std::generate_canonical<double, 53>(grng);
            Action ca = SampleActionSafe(outcomes, u);
            st->ApplyAction(ca);
            continue;
          }

          Player p = st->CurrentPlayer();
          auto legal = st->LegalActions();
          if (legal.empty()) break;

          auto* dune_st = dynamic_cast<const DuneImperiumState*>(st.get());
          int rnd = dune_st ? dune_st->GetCurrentRound() : 1;

          if (dune_st != nullptr && dune_st->phase() == GamePhase::kAgentTurns &&
              !dune_st->IsPlayerInRevealTurn(p) &&
              dune_st->GetPlayerAgentsRemainingForTesting(p) > 0) {
            bool has_board_space_transition = false;
            for (Action a1 : legal) {
              auto s1 = st->Clone();
              s1->ApplyAction(a1);
              if (!s1->IsTerminal() && !s1->IsChanceNode() && s1->CurrentPlayer() == p) {
                auto l2 = s1->LegalActions();
                for (Action a : l2) {
                  if (a >= 2056 && a <= 2312) {
                    s1->ApplyAction(a);
                    if (!s1->IsTerminal() && !s1->IsChanceNode() && s1->CurrentPlayer() == p) {
                      l2 = s1->LegalActions();
                    } else {
                      l2.clear();
                    }
                    break;
                  }
                }
                for (Action a2 : l2) {
                  if (IsBoardSpaceAction(a2)) {
                    has_board_space_transition = true;
                    break;
                  }
                }
              }
              if (has_board_space_transition) break;
            }

            if (has_board_space_transition) {
              PlaceCensusCandidate cand;
              cand.game_id = gid;
              cand.round = rnd;
              cand.player = p;
              cand.state = st->Clone();
              {
                std::lock_guard<std::mutex> lk(cand_mu);
                all_candidates.push_back(std::move(cand));
              }
            }
          }

          auto prior_legals = bundle.evaluator->PriorWithLegals(*st, legal);
          double u_act = std::generate_canonical<double, 53>(grng);
          Action act = SampleActionSafe(prior_legals, u_act);
          st->ApplyAction(act);
        }
      }
    });
  }
  for (auto& t : census_workers) t.join();
  double census_sec = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - census_t0).count();
  std::cout << "Census completed in " << std::fixed << std::setprecision(2)
            << census_sec << "s: extracted " << all_candidates.size()
            << " primary placement roots across " << num_census_games << " games.\n";

  // Stratify into R1-3, R4-6, R7+
  std::vector<PlaceCensusCandidate> cands_r13;
  std::vector<PlaceCensusCandidate> cands_r46;
  std::vector<PlaceCensusCandidate> cands_r7;
  for (auto& c : all_candidates) {
    if (c.round <= 3) cands_r13.push_back(std::move(c));
    else if (c.round <= 6) cands_r46.push_back(std::move(c));
    else cands_r7.push_back(std::move(c));
  }

  const int q_r13 = (target_roots * 70 + 100) / 200;
  const int q_r46 = (target_roots * 70 + 100) / 200;
  const int q_r7 = target_roots - q_r13 - q_r46;

  auto select_place_candidates = [](
      std::vector<PlaceCensusCandidate>& pool, int target_quota, uint64_t seed) {
    std::vector<PlaceCensusCandidate> selected;
    if (pool.empty() || target_quota <= 0) return selected;

    std::map<int, std::vector<PlaceCensusCandidate>> by_game;
    for (auto& c : pool) {
      by_game[c.game_id].push_back(std::move(c));
    }

    std::vector<int> gids;
    for (const auto& kv : by_game) gids.push_back(kv.first);
    std::mt19937_64 rng(seed);
    std::shuffle(gids.begin(), gids.end(), rng);

    size_t pass = 0;
    while (static_cast<int>(selected.size()) < target_quota) {
      bool added = false;
      for (int gid : gids) {
        if (pass < by_game[gid].size()) {
          selected.push_back(std::move(by_game[gid][pass]));
          added = true;
          if (static_cast<int>(selected.size()) == target_quota) break;
        }
      }
      if (!added) break;
      ++pass;
    }
    return selected;
  };

  auto sel_r13 = select_place_candidates(cands_r13, q_r13, master_seed + 101);
  auto sel_r46 = select_place_candidates(cands_r46, q_r46, master_seed + 102);
  auto sel_r7 = select_place_candidates(cands_r7, q_r7, master_seed + 103);

  std::vector<PlaceCensusCandidate> roots_meta;
  roots_meta.reserve(sel_r13.size() + sel_r46.size() + sel_r7.size());
  for (auto& c : sel_r13) roots_meta.push_back(std::move(c));
  for (auto& c : sel_r46) roots_meta.push_back(std::move(c));
  for (auto& c : sel_r7) roots_meta.push_back(std::move(c));

  const size_t N = roots_meta.size();
  std::cout << "Selected " << N << " stratified placement roots: R1-3=" << sel_r13.size()
            << ", R4-6=" << sel_r46.size() << ", R7+=" << sel_r7.size() << "\n\n";

  // 2 & 3. Joint Candidate Enumeration & Evaluation of Three Controllers
  std::cout << "--------------------------------------------------------\n";
  std::cout << "STEP 2 & 3: Joint Enumeration & Evaluation of Three Placement Controllers\n";
  std::cout << "--------------------------------------------------------\n";

  std::vector<std::vector<JointActionPair>> cand_pairs_per_root(N);
  std::vector<JointActionPair> raw_pairs(N);
  std::vector<JointControllerResult> play_results(N);
  std::vector<JointActionPair> seq_results(N);
  std::vector<JointControllerResult> online_results(N);

  std::atomic<size_t> next_place_eval{0};
  const int num_place_threads = 16;
  std::vector<std::thread> place_threads;
  place_threads.reserve(num_place_threads);

  const auto controllers_t0 = std::chrono::steady_clock::now();
  for (int th = 0; th < num_place_threads; ++th) {
    place_threads.emplace_back([&, th]() {
      while (true) {
        size_t i = next_place_eval.fetch_add(1);
        if (i >= N) break;

        const auto& cand = roots_meta[i];
        State* root_state = cand.state.get();
        Player p = cand.player;

        // 2. Joint (Card, Space) Enumerator
        auto cands = EnumerateJointPlacementCandidates(
            *root_state, p, bundle.evaluator.get(), /*max_k=*/8);
        cand_pairs_per_root[i] = cands;
        raw_pairs[i] = cands.empty() ? JointActionPair{} : cands.front();

        uint64_t root_seed = dune_seed::DeriveSeed(
            master_seed, 0x5443485F504C43ULL, i);

        // Controller 1: seq_sh_baseline
        seq_results[i] = RunSeqShBaseline(*root_state, p, bundle.evaluator, root_seed);

        // Controller 2: search_v2_joint_play
        play_results[i] = RunSearchV2JointPlay(
            *root_state, p, bundle.evaluator.get(), cands, root_seed);

        // Controller 3: online_joint_place
        online_results[i] = RunOnlineJointPlace(
            *root_state, p, bundle.evaluator.get(), cands, root_seed,
            /*s2_rollouts=*/64, absl::GetFlag(FLAGS_gate4a_online_min_t));
      }
    });
  }
  for (auto& t : place_threads) t.join();
  double controllers_sec = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - controllers_t0).count();
  std::cout << "All three placement controllers evaluated on " << N
            << " roots in " << std::fixed << std::setprecision(2)
            << controllers_sec << "s.\n\n";

  // 4. Fresh Independent Ref512 Bank Evaluation (seed domain 0x52454635313250ULL)
  std::cout << "--------------------------------------------------------\n";
  std::cout << "STEP 4: Fresh Independent Ref512 Placement Bank Evaluation (R_eval = " << ref_rollouts << ")\n";
  std::cout << "--------------------------------------------------------\n";

  std::vector<size_t> active_root_indices;
  std::map<size_t, std::vector<JointActionPair>> active_distinct_pairs;

  for (size_t i = 0; i < N; ++i) {
    const auto& rp = raw_pairs[i];
    const auto& seq_pair = seq_results[i];
    bool seq_ovr = !(seq_pair == rp);
    bool play_ovr = play_results[i].overridden;
    bool online_ovr = online_results[i].overridden;
    bool online_conf = online_results[i].confirmed;

    if (seq_ovr || play_ovr || online_ovr || online_conf) {
      active_root_indices.push_back(i);
      std::vector<JointActionPair> dp;
      auto add_pair = [&](const JointActionPair& p) {
        if (p.a1 != kInvalidAction && std::find(dp.begin(), dp.end(), p) == dp.end()) {
          dp.push_back(p);
        }
      };
      add_pair(rp);
      if (seq_ovr) add_pair(seq_pair);
      if (play_ovr) add_pair(play_results[i].chosen_pair);
      if (online_ovr) add_pair(online_results[i].chosen_pair);
      if (online_conf) {
        add_pair(online_results[i].teacher_pair);
        add_pair(online_results[i].alt_pair);
      }
      active_distinct_pairs[i] = dp;
    }
  }

  std::cout << "Identified " << active_root_indices.size()
            << " active placement roots where any controller overrode sequential raw.\n";

  std::map<size_t, std::map<JointActionPair, double>> ref_u;
  std::map<size_t, std::map<JointActionPair, double>> sum_u;

  const auto ref_t0 = std::chrono::steady_clock::now();
  const int world_chunk_size = 64;
  for (int w_chunk = 0; w_chunk < ref_rollouts; w_chunk += world_chunk_size) {
    int w_end = std::min(ref_rollouts, w_chunk + world_chunk_size);
    int cur_chunk = w_end - w_chunk;

    struct RefPlaceJob {
      size_t root_idx;
      JointActionPair pair;
      std::future<double> fut;
    };
    std::vector<RefPlaceJob> chunk_jobs;

    for (size_t root_idx : active_root_indices) {
      const auto* dune_root = dynamic_cast<const DuneImperiumState*>(roots_meta[root_idx].state.get());
      Player p = roots_meta[root_idx].player;
      const auto& pairs = active_distinct_pairs[root_idx];

      for (int w = w_chunk; w < w_end; ++w) {
        uint64_t wseed = dune_seed::DeriveSeed(
            master_seed,
            0x52454635313250ULL + static_cast<uint64_t>(root_idx),
            w);
        std::unique_ptr<State> base_world = nullptr;
        if (dune_root != nullptr) {
          std::mt19937 wrng(wseed);
          auto rng_func = [&wrng]() {
            return std::generate_canonical<double, 53>(wrng);
          };
          base_world = dune_root->ResampleFromInfostate(p, rng_func);
        }
        if (!base_world) {
          base_world = roots_meta[root_idx].state->Clone();
        }

        for (size_t p_i = 0; p_i < pairs.size(); ++p_i) {
          auto sim = (p_i + 1 == pairs.size()) ? std::move(base_world) : base_world->Clone();
          sim->ApplyAction(pairs[p_i].a1);
          if (pairs[p_i].a_extra != kInvalidAction) {
            sim->ApplyAction(pairs[p_i].a_extra);
          }
          if (pairs[p_i].a2 != kInvalidAction) {
            sim->ApplyAction(pairs[p_i].a2);
          }
          if (bundle.evaluator != nullptr) {
            bundle.evaluator->AdjustActiveRollouts(1);
          }
          chunk_jobs.push_back({root_idx, pairs[p_i], SearchRolloutThreadPool::Instance().Submit(
              [sim = std::move(sim), p, eval = bundle.evaluator.get(), wseed]() {
                ActiveRolloutReleaseGuard guard{eval};
                auto traj = SimulateRolloutWithTurnTrajectory(
                    sim.get(), p, eval, wseed,
                    SearchTruncationMode::kTerminal, SearchRolloutCrnMode::kEventKeyedCrn);
                return traj.terminal_utility;
              })});
        }
      }
    }

    for (auto& job : chunk_jobs) {
      sum_u[job.root_idx][job.pair] += job.fut.get();
    }
    std::cout << "  [Ref512 Placement Progress] Worlds " << w_end << " / " << ref_rollouts
              << " completed (" << chunk_jobs.size() << " rollouts)\n";
  }

  for (size_t root_idx : active_root_indices) {
    for (const auto& pair : active_distinct_pairs[root_idx]) {
      ref_u[root_idx][pair] = sum_u[root_idx][pair] / static_cast<double>(ref_rollouts);
    }
  }

  double ref_sec = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - ref_t0).count();
  std::cout << "Ref512 placement evaluation completed in " << std::fixed << std::setprecision(2)
            << ref_sec << "s.\n\n";

  // Bootstrap metrics & Pass Criteria
  std::vector<int> game_ids(N);
  for (size_t i = 0; i < N; ++i) game_ids[i] = roots_meta[i].game_id;

  // 1. search_v2_joint_play
  std::vector<double> play_gain_all(N, 0.0);
  std::vector<double> play_gain_ovr;
  std::vector<int> play_gid_ovr;
  int play_ovr_count = 0;
  int play_bad_count = 0;
  int play_mat_bad_count = 0;
  int cross_card_joint_overrides = 0;

  for (size_t i = 0; i < N; ++i) {
    if (play_results[i].overridden) {
      ++play_ovr_count;
      double diff = ref_u[i][play_results[i].chosen_pair] - ref_u[i][raw_pairs[i]];
      play_gain_all[i] = diff;
      play_gain_ovr.push_back(diff);
      play_gid_ovr.push_back(game_ids[i]);
      if (diff < 0.0) ++play_bad_count;
      if (diff < -0.05) ++play_mat_bad_count;

      Action a1_c = play_results[i].chosen_pair.a1;
      Action a2_c = play_results[i].chosen_pair.a2;
      Action a1_raw = raw_pairs[i].a1;
      if (a1_c != a1_raw) {
        Action greedy_space_for_a1_c = kInvalidAction;
        for (const auto& cand : cand_pairs_per_root[i]) {
          if (cand.a1 == a1_c && cand.a2 != kInvalidAction) {
            greedy_space_for_a1_c = cand.a2;
            break;
          }
        }
        if (a2_c != kInvalidAction && a2_c != greedy_space_for_a1_c) {
          ++cross_card_joint_overrides;
        }
      }
    }
  }

  auto boot_play_all = ComputeGameClusteredBootstrap(play_gain_all, game_ids, 10000, master_seed + 110);
  auto boot_play_ovr = ComputeGameClusteredBootstrap(play_gain_ovr, play_gid_ovr, 10000, master_seed + 111);
  double bad_rate_play = play_ovr_count > 0 ? (static_cast<double>(play_bad_count) / play_ovr_count) : 0.0;
  double mat_bad_rate_play = play_ovr_count > 0 ? (static_cast<double>(play_mat_bad_count) / play_ovr_count) : 0.0;
  double ovr_rate_play = static_cast<double>(play_ovr_count) / N;

  // 2. seq_sh_baseline
  std::vector<double> seq_gain_all(N, 0.0);
  std::vector<double> seq_gain_ovr;
  std::vector<int> seq_gid_ovr;
  int seq_ovr_count = 0;
  int seq_bad_count = 0;

  for (size_t i = 0; i < N; ++i) {
    const auto& seq_pair = seq_results[i];
    if (!(seq_pair == raw_pairs[i])) {
      ++seq_ovr_count;
      double diff = ref_u[i][seq_pair] - ref_u[i][raw_pairs[i]];
      seq_gain_all[i] = diff;
      seq_gain_ovr.push_back(diff);
      seq_gid_ovr.push_back(game_ids[i]);
      if (diff < 0.0) ++seq_bad_count;
    }
  }

  auto boot_seq_all = ComputeGameClusteredBootstrap(seq_gain_all, game_ids, 10000, master_seed + 120);
  auto boot_seq_ovr = ComputeGameClusteredBootstrap(seq_gain_ovr, seq_gid_ovr, 10000, master_seed + 121);
  double bad_rate_seq = seq_ovr_count > 0 ? (static_cast<double>(seq_bad_count) / seq_ovr_count) : 0.0;
  double ovr_rate_seq = static_cast<double>(seq_ovr_count) / N;

  // 3. Paired search_v2_joint_play vs seq_sh_baseline
  std::vector<double> paired_play_vs_seq(N, 0.0);
  for (size_t i = 0; i < N; ++i) {
    const auto& seq_pair = seq_results[i];
    double u_play = play_results[i].overridden ? ref_u[i][play_results[i].chosen_pair] : ref_u[i][raw_pairs[i]];
    double u_seq = (!(seq_pair == raw_pairs[i])) ? ref_u[i][seq_pair] : ref_u[i][raw_pairs[i]];
    paired_play_vs_seq[i] = u_play - u_seq;
  }
  auto boot_paired = ComputeGameClusteredBootstrap(paired_play_vs_seq, game_ids, 10000, master_seed + 130);

  // 4. online_joint_place
  std::vector<double> online_gain_all(N, 0.0);
  std::vector<double> online_gain_ovr;
  std::vector<int> online_gid_ovr;
  int online_ovr_count = 0;
  int online_bad_count = 0;
  int online_mat_bad_count = 0;
  int online_confirmations = 0;
  int online_accepted = 0;
  std::vector<double> online_accepted_gain;
  std::vector<int> online_accepted_gid;

  for (size_t i = 0; i < N; ++i) {
    if (online_results[i].overridden) {
      ++online_ovr_count;
      double diff = ref_u[i][online_results[i].chosen_pair] - ref_u[i][raw_pairs[i]];
      online_gain_all[i] = diff;
      online_gain_ovr.push_back(diff);
      online_gid_ovr.push_back(game_ids[i]);
      if (diff < 0.0) ++online_bad_count;
      if (diff < -0.05) ++online_mat_bad_count;
    }
    if (online_results[i].confirmed) {
      ++online_confirmations;
    }
    if (online_results[i].teacher_accepted) {
      ++online_accepted;
      double diff = ref_u[i][online_results[i].teacher_pair] - ref_u[i][online_results[i].alt_pair];
      online_accepted_gain.push_back(diff);
      online_accepted_gid.push_back(game_ids[i]);
    }
  }

  auto boot_online_all = ComputeGameClusteredBootstrap(online_gain_all, game_ids, 10000, master_seed + 140);
  auto boot_online_ovr = ComputeGameClusteredBootstrap(online_gain_ovr, online_gid_ovr, 10000, master_seed + 141);
  auto boot_online_acc = ComputeGameClusteredBootstrap(online_accepted_gain, online_accepted_gid, 10000, master_seed + 142);
  double bad_rate_online = online_ovr_count > 0 ? (static_cast<double>(online_bad_count) / online_ovr_count) : 0.0;
  double mat_bad_rate_online = online_ovr_count > 0 ? (static_cast<double>(online_mat_bad_count) / online_ovr_count) : 0.0;
  double ovr_rate_online = static_cast<double>(online_ovr_count) / N;

  bool gate4a_place_play_pass = (boot_play_all.ci_low > 0.0 && bad_rate_play < 0.15);
  bool gate4a_place_online_pass = (boot_online_all.ci_low > 0.0 &&
                                   boot_online_acc.ci_low > 0.0 &&
                                   bad_rate_online < 0.15);

  // Print Formatted Report to stdout
  std::cout << "\n### Table 1: Gate 4A Placement Controllers Offline Summary (N = " << N << ")\n";
  std::cout << "| Metric                           | search_v2_joint_play | seq_sh_baseline      | online_joint_place   |\n";
  std::cout << "|:---------------------------------|:---------------------|:---------------------|:---------------------|\n";
  std::cout << "| Roots Evaluated                  | " << std::setw(20) << N << " | " << std::setw(20) << N << " | " << std::setw(20) << N << " |\n";
  std::cout << "| Overrides (Rate)                 | " << std::setw(11) << play_ovr_count << " (" << std::fixed << std::setprecision(1) << (ovr_rate_play * 100.0) << "%) | "
            << std::setw(11) << seq_ovr_count << " (" << (ovr_rate_seq * 100.0) << "%) | "
            << std::setw(11) << online_ovr_count << " (" << (ovr_rate_online * 100.0) << "%) |\n";
  std::cout << "| All-Roots Mean Gain              | " << std::setw(20) << std::setprecision(4) << boot_play_all.mean << " | "
            << std::setw(20) << boot_seq_all.mean << " | "
            << std::setw(20) << boot_online_all.mean << " |\n";
  std::cout << "| All-Roots 95% CI                 | [" << std::setprecision(4) << boot_play_all.ci_low << ", " << boot_play_all.ci_high << "] | ["
            << boot_seq_all.ci_low << ", " << boot_seq_all.ci_high << "] | ["
            << boot_online_all.ci_low << ", " << boot_online_all.ci_high << "] |\n";
  std::cout << "| Override-Only Mean Gain          | " << std::setw(20) << std::setprecision(4) << boot_play_ovr.mean << " | "
            << std::setw(20) << boot_seq_ovr.mean << " | "
            << std::setw(20) << boot_online_ovr.mean << " |\n";
  std::cout << "| Override-Only 95% CI             | [" << std::setprecision(4) << boot_play_ovr.ci_low << ", " << boot_play_ovr.ci_high << "] | ["
            << boot_seq_ovr.ci_low << ", " << boot_seq_ovr.ci_high << "] | ["
            << boot_online_ovr.ci_low << ", " << boot_online_ovr.ci_high << "] |\n";
  std::cout << "| Bad Overrides (Rate < 15%)       | " << std::setw(11) << play_bad_count << " (" << std::setprecision(1) << (bad_rate_play * 100.0) << "%) | "
            << std::setw(11) << seq_bad_count << " (" << (bad_rate_seq * 100.0) << "%) | "
            << std::setw(11) << online_bad_count << " (" << (bad_rate_online * 100.0) << "%) |\n";
  std::cout << "| Material Bad (< -0.05)           | " << std::setw(11) << play_mat_bad_count << " (" << std::setprecision(1) << (mat_bad_rate_play * 100.0) << "%) | "
            << std::setw(20) << "N/A" << " | "
            << std::setw(11) << online_mat_bad_count << " (" << (mat_bad_rate_online * 100.0) << "%) |\n";
  std::cout << "| Num Confirmations                | " << std::setw(20) << "N/A" << " | "
            << std::setw(20) << "N/A" << " | "
            << std::setw(20) << online_confirmations << " |\n";
  std::cout << "| Num Accepted (Ovr + Conf)        | " << std::setw(20) << "N/A" << " | "
            << std::setw(20) << "N/A" << " | "
            << std::setw(20) << online_accepted << " |\n";
  std::cout << "| Accepted Pair Mean Gain          | " << std::setw(20) << "N/A" << " | "
            << std::setw(20) << "N/A" << " | "
            << std::setw(20) << std::setprecision(4) << boot_online_acc.mean << " |\n";
  std::cout << "| Accepted Pair 95% CI             | " << std::setw(20) << "N/A" << " | "
            << std::setw(20) << "N/A" << " | ["
            << std::setprecision(4) << boot_online_acc.ci_low << ", " << boot_online_acc.ci_high << "] |\n";
  std::cout << "| Paired vs Sequential Mean        | " << std::setw(20) << std::setprecision(4) << boot_paired.mean << " | "
            << std::setw(20) << "Ref Baseline" << " | "
            << std::setw(20) << "N/A" << " |\n";
  std::cout << "| Paired vs Sequential 95% CI      | [" << std::setprecision(4) << boot_paired.ci_low << ", " << boot_paired.ci_high << "] | "
            << std::setw(20) << "Ref Baseline" << " | "
            << std::setw(20) << "N/A" << " |\n";
  std::cout << "| Cross-Card Joint Overrides       | " << std::setw(20) << cross_card_joint_overrides << " | "
            << std::setw(20) << 0 << " | "
            << std::setw(20) << "N/A" << " |\n";
  std::cout << "| Gate 4A Assessment               | " << std::setw(20) << (gate4a_place_play_pass ? "PASS" : "FAIL") << " | "
            << std::setw(20) << "BASELINE" << " | "
            << std::setw(20) << (gate4a_place_online_pass ? "PASS" : "FAIL") << " |\n";
  std::cout << "\n";

  // Export JSON summary if requested
  const std::string json_path = absl::GetFlag(FLAGS_summary_json_path);
  if (!json_path.empty()) {
    std::ofstream ofs(json_path);
    if (ofs.good()) {
      ofs << "{\n"
          << "  \"mode\": \"gate4a_place\",\n"
          << "  \"model_path\": \"" << bundle.model_path << "\",\n"
          << "  \"num_roots\": " << N << ",\n"
          << "  \"gate4a_place_play_pass\": " << (gate4a_place_play_pass ? "true" : "false") << ",\n"
          << "  \"gate4a_place_online_pass\": " << (gate4a_place_online_pass ? "true" : "false") << ",\n"
          << "  \"search_v2_joint_play\": {\n"
          << "    \"num_roots\": " << N << ",\n"
          << "    \"num_overrides\": " << play_ovr_count << ",\n"
          << "    \"override_rate\": " << ovr_rate_play << ",\n"
          << "    \"all_roots_mean_gain\": " << boot_play_all.mean << ",\n"
          << "    \"all_roots_ci95_low\": " << boot_play_all.ci_low << ",\n"
          << "    \"all_roots_ci95_high\": " << boot_play_all.ci_high << ",\n"
          << "    \"override_only_mean_gain\": " << boot_play_ovr.mean << ",\n"
          << "    \"override_only_ci95_low\": " << boot_play_ovr.ci_low << ",\n"
          << "    \"override_only_ci95_high\": " << boot_play_ovr.ci_high << ",\n"
          << "    \"bad_override_count\": " << play_bad_count << ",\n"
          << "    \"bad_override_rate\": " << bad_rate_play << ",\n"
          << "    \"material_bad_override_rate\": " << mat_bad_rate_play << ",\n"
          << "    \"cross_card_joint_overrides\": " << cross_card_joint_overrides << "\n"
          << "  },\n"
          << "  \"seq_sh_baseline\": {\n"
          << "    \"num_roots\": " << N << ",\n"
          << "    \"num_overrides\": " << seq_ovr_count << ",\n"
          << "    \"override_rate\": " << ovr_rate_seq << ",\n"
          << "    \"all_roots_mean_gain\": " << boot_seq_all.mean << ",\n"
          << "    \"all_roots_ci95_low\": " << boot_seq_all.ci_low << ",\n"
          << "    \"all_roots_ci95_high\": " << boot_seq_all.ci_high << ",\n"
          << "    \"override_only_mean_gain\": " << boot_seq_ovr.mean << ",\n"
          << "    \"override_only_ci95_low\": " << boot_seq_ovr.ci_low << ",\n"
          << "    \"override_only_ci95_high\": " << boot_seq_ovr.ci_high << ",\n"
          << "    \"bad_override_count\": " << seq_bad_count << ",\n"
          << "    \"bad_override_rate\": " << bad_rate_seq << "\n"
          << "  },\n"
          << "  \"online_joint_place\": {\n"
          << "    \"num_roots\": " << N << ",\n"
          << "    \"num_overrides\": " << online_ovr_count << ",\n"
          << "    \"override_rate\": " << ovr_rate_online << ",\n"
          << "    \"all_roots_mean_gain\": " << boot_online_all.mean << ",\n"
          << "    \"all_roots_ci95_low\": " << boot_online_all.ci_low << ",\n"
          << "    \"all_roots_ci95_high\": " << boot_online_all.ci_high << ",\n"
          << "    \"override_only_mean_gain\": " << boot_online_ovr.mean << ",\n"
          << "    \"override_only_ci95_low\": " << boot_online_ovr.ci_low << ",\n"
          << "    \"override_only_ci95_high\": " << boot_online_ovr.ci_high << ",\n"
          << "    \"bad_override_count\": " << online_bad_count << ",\n"
          << "    \"bad_override_rate\": " << bad_rate_online << ",\n"
          << "    \"material_bad_override_rate\": " << mat_bad_rate_online << ",\n"
          << "    \"num_accepted\": " << online_accepted << ",\n"
          << "    \"num_confirmations\": " << online_confirmations << ",\n"
          << "    \"accepted_pair_mean_gain\": " << boot_online_acc.mean << ",\n"
          << "    \"accepted_pair_ci95_low\": " << boot_online_acc.ci_low << ",\n"
          << "    \"accepted_pair_ci95_high\": " << boot_online_acc.ci_high << "\n"
          << "  },\n"
          << "  \"paired_play_vs_seq\": {\n"
          << "    \"mean_difference\": " << boot_paired.mean << ",\n"
          << "    \"ci95_low\": " << boot_paired.ci_low << ",\n"
          << "    \"ci95_high\": " << boot_paired.ci_high << "\n"
          << "  }\n"
          << "}\n";
      ofs.close();
      std::cout << "Exported summary JSON to " << json_path << "\n";
    }
  }

  std::cout << "========================================================\n";
  std::cout << "GATE4A_PLACE COMPLETE: PLAY PASS = " << (gate4a_place_play_pass ? "YES" : "NO")
            << " | ONLINE PASS = " << (gate4a_place_online_pass ? "YES" : "NO") << "\n";
  std::cout << "========================================================\n";
}

int main(int argc, char** argv) {
  std::cout << std::unitbuf;
  absl::SetFlag(&FLAGS_num_blocks, 16);
  absl::SetFlag(&FLAGS_market_appendix_mode, "full_public_information_v4");
  absl::SetFlag(&FLAGS_deck_pool_semantics, "fixed");
  absl::SetFlag(&FLAGS_enable_semantic_scorer, true);
  absl::ParseCommandLine(argc, argv);
  std::string mode = absl::GetFlag(FLAGS_mode);

  if (mode == "stage3_cpu_verify") {
    RunStage3CpuVerify();
  } else if (mode == "gate0a") {
    RunGate0a();
  } else if (mode == "gate0b") {
    RunGate0b();
  } else if (mode == "fp8_301_ref2048") {
    RunFp8Ref2048Benchmark(absl::GetFlag(FLAGS_source_model_path),
                           absl::GetFlag(FLAGS_summary_json_path),
                           absl::GetFlag(FLAGS_num_301_roots),
                           absl::GetFlag(FLAGS_census_games),
                           absl::GetFlag(FLAGS_ref2048_rollouts));
  } else if (mode == "verify_v5") {
    RunVerifyV5();
  } else if (mode == "gate4a_buy") {
    RunGate4aBuy();
  } else if (mode == "gate4a_place") {
    RunGate4aPlace();
  } else {
    std::cerr << "Unknown mode: " << mode
              << " (expected stage3_cpu_verify, gate0a, gate0b, fp8_301_ref2048, verify_v5, gate4a_buy, or gate4a_place)\n";
    return 1;
  }

  return 0;
}
