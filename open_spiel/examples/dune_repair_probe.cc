#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <iomanip>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <algorithm>

#include "open_spiel/spiel.h"
#include "open_spiel/utils/json.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "dune_network.h"
#include "dune_evaluator.h"
#include "dune_batched_evaluator.h"
#include "dune_puct_is_mcts.h"
#include "dune_seed_utils.h"
#include "dune_sha256.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"

ABSL_FLAG(std::string, model_checkpoint,
          "/run/media/warcr/Storage/dune_drl_runtime/round7/u20328_continuation_20260919_234946/checkpoints/ppo_model_update_21328.pt",
          "Path to U21328 model checkpoint");
ABSL_FLAG(std::string, manifest_path, "retained_roots_manifest.json",
          "Path to 24-root retained manifest JSON");
ABSL_FLAG(std::string, output_jsonl, "retained_roots_probe_results.jsonl",
          "Path to write JSONL probe results");
ABSL_FLAG(std::string, receipt_path, "retained_roots_probe_receipt.json",
          "Path to write durable receipt JSON");
ABSL_FLAG(int, hidden_dim, 2048, "Hidden dim");
ABSL_FLAG(int, num_blocks, 8, "Residual blocks count");

namespace open_spiel {

const char* kExpectedCkptSha256 =
    "dbdc36ac64d776702f4de2cc421e63cd1b590afae7d1ff7d4f2a5f411d261777";

int RunProbe(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  const std::string ckpt_path = absl::GetFlag(FLAGS_model_checkpoint);
  const std::string manifest_path = absl::GetFlag(FLAGS_manifest_path);
  const std::string out_jsonl_path = absl::GetFlag(FLAGS_output_jsonl);

  std::cout << "=== Bounded Retained-Root Native Probe ===\n";
  std::cout << "Checkpoint: " << ckpt_path << "\n";
  std::cout << "Manifest:   " << manifest_path << "\n";
  std::cout << "Output:     " << out_jsonl_path << "\n";

  // 1. Verify Checkpoint SHA-256
  std::string actual_sha = ComputeFileSHA256(ckpt_path);
  std::cout << "Checkpoint SHA-256: " << actual_sha << "\n";
  if (actual_sha != kExpectedCkptSha256) {
    std::cerr << "FATAL: Checkpoint SHA-256 mismatch! Expected "
              << kExpectedCkptSha256 << " but got " << actual_sha << "\n";
    return 1;
  }
  std::cout << "Checkpoint SHA-256 verified successfully.\n";

  std::string manifest_sha256 = ComputeFileSHA256(manifest_path);
  std::cout << "Manifest SHA-256:   " << manifest_sha256 << "\n";

  std::string probe_binary_sha256 = ComputeFileSHA256("/proc/self/exe");
  std::cout << "Probe binary SHA:   " << probe_binary_sha256 << "\n";

  // 2. Load Manifest
  std::ifstream mf_file(manifest_path);
  if (!mf_file.is_open()) {
    std::cerr << "FATAL: Could not open manifest: " << manifest_path << "\n";
    return 1;
  }
  std::string mf_str((std::istreambuf_iterator<char>(mf_file)),
                     std::istreambuf_iterator<char>());
  auto mf_json = json::FromString(mf_str);
  if (!mf_json.has_value() || !mf_json->IsArray()) {
    std::cerr << "FATAL: Invalid manifest JSON\n";
    return 1;
  }
  const auto& roots = mf_json->GetArray();
  const size_t num_roots = roots.size();
  std::cout << "Loaded " << num_roots << " roots from manifest.\n";
  if (num_roots > 24) {
    std::cerr << "FATAL: Manifest exceeds maximum 24 retained roots: "
              << num_roots << "\n";
    return 1;
  }

  // 3. Setup Game and Neural Network Evaluator
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  int64_t model_dim = game->InformationStateTensorSize();
  {
    torch::serialize::InputArchive archive;
    archive.load_from(ckpt_path, torch::kCPU);
    torch::serialize::InputArchive in_archive;
    if (archive.try_read("input_layer", in_archive)) {
      torch::Tensor w;
      if (in_archive.try_read("weight", w)) {
        model_dim = w.size(1);
      }
    }
  }
  int64_t action_size = game->NumDistinctActions();
  bool cuda_available = torch::cuda::is_available();
  torch::Device device = cuda_available
                             ? torch::Device(torch::kCUDA)
                             : torch::Device(torch::kCPU);
  std::string actual_device_str = device.is_cuda() ? "cuda" : "cpu";
  std::string actual_precision_str = device.is_cuda() ? "BF16" : "FP32";
  std::cout << "Torch device: " << device << " (" << actual_device_str
            << "), precision: " << actual_precision_str
            << ", model_dim: " << model_dim
            << ", action_size: " << action_size << "\n";

  const bool has_scorer = CheckpointHasSemanticScorer(ckpt_path, torch::kCPU);
  int hidden_dim = absl::GetFlag(FLAGS_hidden_dim);
  int num_blocks = absl::GetFlag(FLAGS_num_blocks);

  auto search_model = std::make_shared<SharedDunePolicyValueNetImpl>(
      model_dim, hidden_dim, action_size, num_blocks,
      /*nonlinear_value_head=*/false, /*with_aux_heads=*/false,
      /*head_init_seed=*/0, /*with_semantic_scorer=*/has_scorer);
  LoadModelCheckpointRobust(search_model, ckpt_path, device);
  search_model->to(device);
  search_model->eval();
  std::shared_mutex model_mutex;

  // Batched evaluator with BF16 autocast, TF32 disabled, logit cap 10.0
  auto batched_coord = std::make_shared<BatchedEvaluator>(
      search_model, /*target_batch_size=*/16, /*timeout_ms=*/2, device,
      &model_mutex, 10.0f, /*device_synchronize=*/true,
      /*high_priority_stream=*/false, /*emit_batch_membership=*/false,
      /*rollout_amp=*/true, /*allow_tf32=*/false);
  auto evaluator = std::make_shared<BatchedNNEvaluator>(batched_coord, 10.0f);

  std::ofstream out_file(out_jsonl_path);
  if (!out_file.is_open()) {
    std::cerr << "FATAL: Could not open output file: " << out_jsonl_path << "\n";
    return 1;
  }

  // 4. Run Native Probes
  int total_sims_completed = 0;
  bool all_roots_passed = true;
  size_t count_fallback = 0;
  size_t count_bypass = 0;
  size_t count_accepted = 0;

  for (size_t i = 0; i < num_roots; ++i) {
    const auto& r_obj = roots[i].GetObject();
    int probe_idx = r_obj.at("probe_idx").GetInt();
    std::string category = r_obj.at("category").GetString();
    int episode_id = r_obj.at("episode_id").GetInt();
    int decision_ordinal = r_obj.at("decision_ordinal").GetInt();
    uint64_t game_seed = 0;
    auto seed_it = r_obj.find("game_seed");
    if (seed_it != r_obj.end()) {
      game_seed = static_cast<uint64_t>(seed_it->second.GetInt());
    }
    Player recorded_player = static_cast<Player>(r_obj.at("player").GetInt());
    int recorded_round = r_obj.at("round").GetInt();
    std::vector<Action> recorded_legals;
    for (const auto& a_val : r_obj.at("legal_actions").GetArray()) {
      recorded_legals.push_back(static_cast<Action>(a_val.GetInt()));
    }
    std::vector<Action> history;
    for (const auto& a_val : r_obj.at("history").GetArray()) {
      history.push_back(static_cast<Action>(a_val.GetInt()));
    }

    // Reconstruct state
    auto state = game->NewInitialState();
    for (Action a : history) {
      state->ApplyAction(a);
    }

    // Native validation against recorded state
    bool native_valid = true;
    if (state->CurrentPlayer() != recorded_player) {
      std::cerr << "Root " << probe_idx << " player mismatch: got "
                << state->CurrentPlayer() << " exp " << recorded_player << "\n";
      native_valid = false;
    }
    std::vector<Action> state_legals = state->LegalActions();
    std::vector<Action> sorted_state_legals = state_legals;
    std::sort(sorted_state_legals.begin(), sorted_state_legals.end());
    std::vector<Action> sorted_rec_legals = recorded_legals;
    std::sort(sorted_rec_legals.begin(), sorted_rec_legals.end());
    if (sorted_state_legals != sorted_rec_legals) {
      std::cerr << "Root " << probe_idx << " legal actions mismatch!\n";
      native_valid = false;
    }
    const auto* dune =
        dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
    SPIEL_CHECK_TRUE(dune != nullptr);
    if (dune->GetCurrentRound() != recorded_round) {
      std::cerr << "Root " << probe_idx << " round mismatch: got "
                << dune->GetCurrentRound() << " exp " << recorded_round << "\n";
      native_valid = false;
    }

    // Observation validity check
    std::vector<float> obs =
        evaluator->GetConsumedObservation(*state, state->CurrentPlayer());
    if (obs.size() != dune_imperium::kFullPublicInformationStateSize) {
      std::cerr << "Root " << probe_idx << " observation size mismatch!\n";
      native_valid = false;
    }
    for (float f : obs) {
      if (!std::isfinite(f)) {
        std::cerr << "Root " << probe_idx << " non-finite observation value!\n";
        native_valid = false;
        break;
      }
    }

    if (!native_valid) {
      std::cerr << "FATAL: Native verification failed on root " << probe_idx
                << "\n";
      return 1;
    }

    SHA256 obs_hasher;
    obs_hasher.Update(obs.data(), obs.size() * sizeof(float));
    std::string obs_fingerprint = obs_hasher.Final();

    SHA256 state_hasher;
    state_hasher.Update(state->ObservationString());
    std::string state_fingerprint = state_hasher.Final();

    std::string source_ref =
        absl::StrFormat("ep_%d_dec_%d", episode_id, decision_ordinal);

    // Evaluate raw network prior directly on *state
    ActionsAndProbs raw_eval = evaluator->Prior(*state);
    // Legal action argmax using strict > in state->LegalActions() order
    Action direct_raw_argmax = kInvalidAction;
    double direct_max_raw_prob = -1.0;
    absl::flat_hash_map<Action, double> raw_prob_map;
    for (const auto& ap : raw_eval) {
      raw_prob_map[ap.first] = ap.second;
    }
    for (Action a : state_legals) {
      auto it = raw_prob_map.find(a);
      double p = (it != raw_prob_map.end()) ? it->second : 0.0;
      if (p > direct_max_raw_prob) {
        direct_max_raw_prob = p;
        direct_raw_argmax = a;
      }
    }

    // Configure DuneSearchConfig for probe
    DuneSearchConfig cfg;
    cfg.max_simulations = 200;
    cfg.relative_time_budget_ms = std::numeric_limits<double>::infinity();
    cfg.max_nodes = 50000;
    cfg.puct_c = 0.3;
    cfg.opponent_mode = SearchOpponentMode::kMaxN;
    cfg.temperature = 0.0;
    cfg.opponent_temperature = 1.0;
    cfg.utility_divisor = 4.0;
    cfg.min_visit_threshold = 2;
    cfg.covered_prior_threshold = 0.50;
    std::mt19937 game_rng(game_seed);
    uint64_t seat_seed = game_seed;
    for (int p = 0; p < 4; ++p) {
      uint64_t s = game_rng();
      if (p == recorded_player) {
        seat_seed = s;
      }
    }
    cfg.seed = seat_seed;
    cfg.final_policy_type = DuneISMCTSFinalPolicyType::kNormalizedVisitCount;
    cfg.dirichlet_epsilon = 0.0;
    cfg.use_observation_string = true;
    cfg.verbose_diagnostics = false;
    cfg.check_strategic_state = true;
    cfg.root_prior_temperature = 1.0;
    cfg.search_leader_draft = false;
    cfg.leader_mass_only_coverage = false;
    cfg.model_checkpoint_path = ckpt_path;

    // 1. Corrected bot run (action_selection_mode = kPreserveRawGreedy)
    cfg.action_selection_mode = DuneActionSelectionMode::kPreserveRawGreedy;
    DunePUCTISMCTSBot bot_corr(cfg, evaluator);
    bot_corr.SetSearchCount(decision_ordinal - 1);

    Action corr_action = bot_corr.Step(*state);
    const DuneSearchResult corr_res = bot_corr.GetLastSearchResult();
    const auto& corr_diag = corr_res.diagnostics;

    total_sims_completed += corr_res.simulations_completed;
    if (corr_res.simulations_completed > 200) {
      std::cerr << "FATAL: Root " << probe_idx << " exceeded 200 simulations: "
                << corr_res.simulations_completed << "\n";
      return 1;
    }

    // 2. Selection on IDENTICAL search result under kHistoricalSample
    cfg.action_selection_mode = DuneActionSelectionMode::kHistoricalSample;
    DunePUCTISMCTSBot bot_hist_selector(cfg, evaluator);
    bot_hist_selector.SetSearchCount(decision_ordinal - 1);
    auto hist_sel = bot_hist_selector.SelectAction(*state, corr_res);

    // 3. Historical bot entry point run
    DunePUCTISMCTSBot bot_hist(cfg, evaluator);
    bot_hist.SetSearchCount(decision_ordinal - 1);
    Action hist_bot_action = bot_hist.Step(*state);
    const DuneSearchResult hist_res = bot_hist.GetLastSearchResult();

    total_sims_completed += hist_res.simulations_completed;
    if (hist_res.simulations_completed > 200) {
      std::cerr << "FATAL: Historical root " << probe_idx
                << " exceeded 200 simulations: "
                << hist_res.simulations_completed << "\n";
      return 1;
    }
    if (total_sims_completed > 10000) {
      std::cerr << "FATAL: Total simulations exceeded 10,000 budget: "
                << total_sims_completed << "\n";
      return 1;
    }

    // Verification checks
    bool incumbent_pass = true;
    bool accepted_search_pass = true;

    if (!corr_diag.search_accepted) {
      // Bypassed or fallback: MUST exactly equal incumbent action
      if (corr_action != corr_diag.incumbent_action) {
        std::cerr << "FAIL: Root " << probe_idx << " (" << category
                  << ") selected " << corr_action << " != incumbent "
                  << corr_diag.incumbent_action << "\n";
        incumbent_pass = false;
        all_roots_passed = false;
      }
      if (corr_action != direct_raw_argmax) {
        std::cerr << "FAIL: Root " << probe_idx << " (" << category
                  << ") selected " << corr_action << " != direct raw argmax "
                  << direct_raw_argmax << "\n";
        incumbent_pass = false;
        all_roots_passed = false;
      }
      if (corr_diag.raw_prior_selected_prob != corr_diag.raw_prior_max_prob) {
        std::cerr << "FAIL: Root " << probe_idx << " (" << category
                  << ") selected prob != max prob!\n";
        incumbent_pass = false;
        all_roots_passed = false;
      }
      if (corr_diag.raw_prior_selected_prob_hex !=
          corr_diag.raw_prior_max_prob_hex) {
        std::cerr << "FAIL: Root " << probe_idx << " (" << category
                  << ") selected prob hex != max prob hex!\n";
        incumbent_pass = false;
        all_roots_passed = false;
      }
      if (category == "low_coverage") count_fallback++;
      if (category == "non_strategic_state") count_bypass++;
    } else {
      // Accepted search: action MUST be preserved identically to historical selection
      // on identical search result
      if (corr_action != hist_sel.selected_action) {
        std::cerr << "FAIL: Root " << probe_idx << " (accepted_search) corr "
                  << corr_action << " != hist on identical result "
                  << hist_sel.selected_action << "\n";
        accepted_search_pass = false;
        all_roots_passed = false;
      }
      count_accepted++;
    }

    // Output JSONL record
    json::Object rec;
    rec["probe_idx"] = static_cast<int64_t>(probe_idx);
    rec["category"] = category;
    rec["episode_id"] = static_cast<int64_t>(episode_id);
    rec["decision_ordinal"] = static_cast<int64_t>(decision_ordinal);
    rec["player"] = static_cast<int64_t>(recorded_player);
    rec["round"] = static_cast<int64_t>(recorded_round);
    rec["legal_action_count"] = static_cast<int64_t>(state_legals.size());

    json::Array legals_arr;
    for (Action a : state_legals) legals_arr.push_back(static_cast<int64_t>(a));
    rec["legal_actions"] = legals_arr;

    rec["native_validation_passed"] = true;
    rec["direct_raw_argmax"] = static_cast<int64_t>(direct_raw_argmax);
    rec["direct_raw_max_prob"] = direct_max_raw_prob;
    rec["direct_raw_max_prob_hex"] = absl::StrFormat("%a", direct_max_raw_prob);

    rec["corr_selected_action"] = static_cast<int64_t>(corr_action);
    rec["corr_incumbent_action"] = static_cast<int64_t>(corr_diag.incumbent_action);
    rec["corr_selection_mode"] = corr_diag.selection_mode;
    rec["corr_selection_reason"] = corr_diag.selection_reason;
    rec["corr_search_accepted"] = corr_diag.search_accepted;
    rec["corr_fallback_reason"] = corr_res.fallback_reason;
    rec["corr_used_fallback"] = corr_res.used_fallback;
    rec["corr_simulations_completed"] = static_cast<int64_t>(corr_res.simulations_completed);
    rec["corr_elapsed_time_ms"] = corr_res.elapsed_time_ms;
    rec["corr_raw_prior_max_prob"] = corr_diag.raw_prior_max_prob;
    rec["corr_raw_prior_selected_prob"] = corr_diag.raw_prior_selected_prob;
    rec["corr_raw_prior_max_prob_hex"] = corr_diag.raw_prior_max_prob_hex;
    rec["corr_raw_prior_selected_prob_hex"] = corr_diag.raw_prior_selected_prob_hex;

    json::Array raw_priors_hex_arr;
    for (double p : corr_diag.raw_priors) {
      raw_priors_hex_arr.push_back(absl::StrFormat("%a", p));
    }
    rec["corr_raw_priors_hex"] = raw_priors_hex_arr;

    rec["hist_identical_search_selected_action"] =
        static_cast<int64_t>(hist_sel.selected_action);
    rec["hist_bot_selected_action"] = static_cast<int64_t>(hist_bot_action);
    rec["hist_bot_simulations_completed"] =
        static_cast<int64_t>(hist_res.simulations_completed);

    rec["manifest_sha256"] = manifest_sha256;
    rec["checkpoint_sha256"] = actual_sha;
    rec["probe_binary_sha256"] = probe_binary_sha256;
    rec["device"] = actual_device_str;
    rec["precision"] = actual_precision_str;
    rec["observation_fingerprint"] = obs_fingerprint;
    rec["state_fingerprint"] = state_fingerprint;
    rec["source_reference"] = source_ref;

    rec["incumbent_agreement_passed"] = incumbent_pass;
    rec["accepted_search_agreement_passed"] = accepted_search_pass;

    out_file << json::ToString(rec, false) << "\n";
    out_file.flush();

    std::cout << absl::StrFormat(
        "[%2zu/24] [%-19s] p=%d act=%-5d inc=%-5d match=%d sims=%-3d reason=%s\n",
        i + 1, category, recorded_player, corr_action,
        corr_diag.incumbent_action,
        (corr_action == corr_diag.incumbent_action ? 1 : 0),
        corr_res.simulations_completed, corr_diag.selection_reason);
  }

  std::cout << "\n=== Retained-Root Probe Summary ===\n";
  std::cout << "Total roots probed: " << num_roots << "\n";
  std::cout << "  - Non-strategic bypass roots: " << count_bypass << " / 8\n";
  std::cout << "  - Low-coverage fallback roots: " << count_fallback << " / 8\n";
  std::cout << "  - Accepted search roots:       " << count_accepted << " / 8\n";
  std::cout << "Total simulations performed:   " << total_sims_completed
            << " (budget ceiling: 10000)\n";
  std::cout << "All roots passed criteria:     "
            << (all_roots_passed ? "PASS" : "FAIL") << "\n";

  if (total_sims_completed > 10000) {
    std::cerr << "FATAL: Total simulations exceeded 10,000 ceiling: "
              << total_sims_completed << "\n";
    return 1;
  }

  // Write durable receipt JSON
  const std::string receipt_path = absl::GetFlag(FLAGS_receipt_path);
  std::ofstream receipt_file(receipt_path);
  if (receipt_file.is_open()) {
    json::Object receipt;
    receipt["checkpoint_path"] = ckpt_path;
    receipt["checkpoint_sha256"] = actual_sha;
    receipt["manifest_path"] = manifest_path;
    receipt["manifest_sha256"] = manifest_sha256;
    receipt["probe_binary_path"] = "/proc/self/exe";
    receipt["probe_binary_sha256"] = probe_binary_sha256;
    receipt["device"] = actual_device_str;
    receipt["precision"] = actual_precision_str;
    receipt["cuda_available"] = cuda_available;
    receipt["total_roots"] = static_cast<int64_t>(num_roots);
    receipt["counts_by_category"] = json::Object{
        {"non_strategic_state", static_cast<int64_t>(count_bypass)},
        {"low_coverage", static_cast<int64_t>(count_fallback)},
        {"accepted_search", static_cast<int64_t>(count_accepted)}};
    receipt["effective_search_settings"] = json::Object{
        {"max_simulations", static_cast<int64_t>(200)},
        {"puct_c", 0.3},
        {"min_visit_threshold", static_cast<int64_t>(2)},
        {"covered_prior_threshold", 0.50},
        {"utility_divisor", 4.0},
        {"temperature", 0.0},
        {"action_selection_mode", "preserve_raw_greedy"}};
    receipt["simulations_completed"] = json::Object{
        {"corrected_simulations", static_cast<int64_t>(count_accepted * 200 + count_fallback * 200)},
        {"historical_simulations", static_cast<int64_t>(count_accepted * 200 + count_fallback * 200)},
        {"total_simulations", static_cast<int64_t>(total_sims_completed)},
        {"budget_ceiling", static_cast<int64_t>(10000)}};
    receipt["all_criteria_passed"] = all_roots_passed;
    receipt_file << json::ToString(receipt, true) << "\n";
    std::cout << "Wrote durable receipt to " << receipt_path << "\n";
  }

  return all_roots_passed ? 0 : 1;
}

}  // namespace open_spiel

int main(int argc, char* argv[]) {
  return open_spiel::RunProbe(argc, argv);
}
