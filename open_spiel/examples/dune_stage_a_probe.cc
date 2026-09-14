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
#include <condition_variable>

#include "open_spiel/spiel.h"
#include "open_spiel/utils/json.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "dune_network.h"
#include "dune_evaluator.h"
#include "dune_batched_evaluator.h"
#include "dune_semantic_action_scorer.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"

ABSL_FLAG(std::string, model_checkpoint, "", "Path to B5300 checkpoint");
ABSL_FLAG(std::string, manifest_path, "", "Path to frozen 16-position manifest");
ABSL_FLAG(std::string, output_json_path, "", "Path to write stage A results JSON");
ABSL_FLAG(int, hidden_dim, 2048, "Hidden dim");
ABSL_FLAG(int, num_blocks, 8, "Residual blocks count");

namespace open_spiel {

struct SimpleBarrier {
  std::mutex mtx;
  std::condition_variable cv;
  int count;
  int total;
  int generation{0};

  SimpleBarrier(int n) : count(n), total(n) {}

  void wait() {
    std::unique_lock<std::mutex> lock(mtx);
    int gen = generation;
    if (--count == 0) {
      generation++;
      count = total;
      cv.notify_all();
    } else {
      cv.wait(lock, [&]() { return generation != gen; });
    }
  }
};

int RunProbe(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  const std::string ckpt_path = absl::GetFlag(FLAGS_model_checkpoint);
  const std::string manifest_path = absl::GetFlag(FLAGS_manifest_path);
  const std::string out_json_path = absl::GetFlag(FLAGS_output_json_path);

  if (ckpt_path.empty() || manifest_path.empty()) {
    std::cerr << "Usage: dune_stage_a_probe --model_checkpoint=<pt> --manifest_path=<json> [--output_json_path=<json>]\n";
    return 1;
  }

  std::cout << "=== Stage A Native Evaluator Diagnostic Probe ===\n";
  std::cout << "Checkpoint: " << ckpt_path << "\n";
  std::cout << "Manifest:   " << manifest_path << "\n";

  // Load manifest
  std::ifstream mf_file(manifest_path);
  if (!mf_file.is_open()) {
    std::cerr << "Error: Could not open manifest: " << manifest_path << "\n";
    return 1;
  }
  std::string mf_str((std::istreambuf_iterator<char>(mf_file)), std::istreambuf_iterator<char>());
  auto mf_json = json::FromString(mf_str);
  if (!mf_json.has_value() || !mf_json->IsArray()) {
    std::cerr << "Error: Invalid manifest JSON\n";
    return 1;
  }
  const auto& positions = mf_json->GetArray();
  const size_t N_POS = positions.size();
  std::cout << "Loaded " << N_POS << " positions from manifest.\n";

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
  std::cout << "Model input dim: " << model_dim << "\n";
  int64_t action_size = game->NumDistinctActions();
  torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
  std::cout << "Torch device: " << device << "\n";

  // Load Model
  const bool has_scorer = CheckpointHasSemanticScorer(ckpt_path, torch::kCPU);
  int hidden_dim = absl::GetFlag(FLAGS_hidden_dim);
  int num_blocks = absl::GetFlag(FLAGS_num_blocks);

  auto search_model = std::make_shared<SharedDunePolicyValueNetImpl>(
      model_dim, hidden_dim, action_size, num_blocks, /*nonlinear_value_head=*/false,
      /*with_aux_heads=*/false, /*head_init_seed=*/0,
      /*with_semantic_scorer=*/has_scorer);
  LoadModelCheckpointRobust(search_model, ckpt_path, device);
  search_model->to(device);
  search_model->eval();
  std::shared_mutex model_mutex;

  // Evaluators setup
  auto d_bf16 = std::make_shared<DuneNNEvaluator>(
      search_model, device, 10.0f, dune_imperium::MarketAppendixMode::kFullPublicInformationV3,
      /*rollout_amp=*/true, /*allow_tf32=*/false);

  auto d_fp32 = std::make_shared<DuneNNEvaluator>(
      search_model, device, 10.0f, dune_imperium::MarketAppendixMode::kFullPublicInformationV3,
      /*rollout_amp=*/false, /*allow_tf32=*/false);

  auto b1_bf16_coord = std::make_shared<BatchedEvaluator>(
      search_model, /*target_batch_size=*/4, /*timeout_ms=*/0, device, &model_mutex,
      10.0f, /*device_synchronize=*/true, /*high_priority_stream=*/false, /*emit_batch_membership=*/false,
      /*rollout_amp=*/true, /*allow_tf32=*/false);
  auto b1_bf16 = std::make_shared<BatchedNNEvaluator>(b1_bf16_coord, 10.0f);

  auto b1_fp32_coord = std::make_shared<BatchedEvaluator>(
      search_model, /*target_batch_size=*/4, /*timeout_ms=*/0, device, &model_mutex,
      10.0f, /*device_synchronize=*/true, /*high_priority_stream=*/false, /*emit_batch_membership=*/false,
      /*rollout_amp=*/false, /*allow_tf32=*/false);
  auto b1_fp32 = std::make_shared<BatchedNNEvaluator>(b1_fp32_coord, 10.0f);

  auto b16_bf16_coord = std::make_shared<BatchedEvaluator>(
      search_model, /*target_batch_size=*/16, /*timeout_ms=*/10, device, &model_mutex,
      10.0f, /*device_synchronize=*/true, /*high_priority_stream=*/false, /*emit_batch_membership=*/true,
      /*rollout_amp=*/true, /*allow_tf32=*/false);
  b16_bf16_coord->EnableBatcherTelemetry();
  auto b16_bf16 = std::make_shared<BatchedNNEvaluator>(b16_bf16_coord, 10.0f);

  auto b16_fp32_coord = std::make_shared<BatchedEvaluator>(
      search_model, /*target_batch_size=*/16, /*timeout_ms=*/10, device, &model_mutex,
      10.0f, /*device_synchronize=*/true, /*high_priority_stream=*/false, /*emit_batch_membership=*/true,
      /*rollout_amp=*/false, /*allow_tf32=*/false);
  b16_fp32_coord->EnableBatcherTelemetry();
  auto b16_fp32 = std::make_shared<BatchedNNEvaluator>(b16_fp32_coord, 10.0f);

  // Reconstruct states and extract observations & descriptors for all positions
  std::vector<std::shared_ptr<State>> states(N_POS);
  std::vector<int> expected_players(N_POS);
  std::vector<int> expected_rounds(N_POS);
  std::vector<std::string> expected_phases(N_POS);
  std::vector<std::string> categories(N_POS);
  std::vector<std::vector<Action>> expected_legals(N_POS);
  std::vector<std::vector<float>> obs_vectors(N_POS);
  std::vector<dune_semantic::CandidateActionData> cand_data_list(N_POS);

  bool all_mechanical_checks_pass = true;
  bool all_consumed_obs_match = true;

  for (size_t i = 0; i < N_POS; ++i) {
    const auto& pos_obj = positions[i].GetObject();
    int manifest_id = static_cast<int>(pos_obj.at("manifest_id").GetInt());
    categories[i] = pos_obj.at("category").GetString();
    expected_players[i] = static_cast<int>(pos_obj.at("player").GetInt());
    if (pos_obj.find("round") != pos_obj.end()) {
      expected_rounds[i] = static_cast<int>(pos_obj.at("round").GetInt());
    }
    if (pos_obj.find("phase") != pos_obj.end()) {
      expected_phases[i] = pos_obj.at("phase").GetString();
    }
    for (const auto& a_val : pos_obj.at("legal_actions").GetArray()) {
      expected_legals[i].push_back(static_cast<Action>(a_val.GetInt()));
    }

    auto state = game->NewInitialState();
    for (const auto& a_val : pos_obj.at("state_history").GetArray()) {
      state->ApplyAction(static_cast<Action>(a_val.GetInt()));
    }

    // Check player and legal actions
    if (state->CurrentPlayer() != expected_players[i]) {
      std::cerr << "FAIL: Player mismatch at pos " << manifest_id << ": got "
                << state->CurrentPlayer() << " vs exp " << expected_players[i] << "\n";
      all_mechanical_checks_pass = false;
    }
    std::vector<Action> actual_legals = state->LegalActions();
    if (actual_legals != expected_legals[i]) {
      std::cerr << "FAIL: Legal actions mismatch at pos " << manifest_id << "\n";
      all_mechanical_checks_pass = false;
    }

    const auto* dune = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
    SPIEL_CHECK_TRUE(dune != nullptr);

    // Check round
    if (expected_rounds[i] > 0 && dune->GetCurrentRound() != expected_rounds[i]) {
      std::cerr << "FAIL: Round mismatch at pos " << manifest_id << ": got "
                << dune->GetCurrentRound() << " vs exp " << expected_rounds[i] << "\n";
      all_mechanical_checks_pass = false;
    }

    // Check phase
    std::string actual_phase;
    switch (dune->phase()) {
      case dune_imperium::GamePhase::kAgentTurns: actual_phase = "kAgentTurns"; break;
      case dune_imperium::GamePhase::kRevealTurns: actual_phase = "kRevealTurns"; break;
      case dune_imperium::GamePhase::kCombat: actual_phase = "kCombat"; break;
      case dune_imperium::GamePhase::kRoundStart: actual_phase = "kRoundStart"; break;
      case dune_imperium::GamePhase::kLeaderDraft: actual_phase = "kLeaderDraft"; break;
      default: actual_phase = "other"; break;
    }
    if (!expected_phases[i].empty() && actual_phase != expected_phases[i]) {
      std::cerr << "FAIL: Phase mismatch at pos " << manifest_id << ": got "
                << actual_phase << " vs exp " << expected_phases[i] << "\n";
      all_mechanical_checks_pass = false;
    }

    // Input witness: Compare consumed observations across ALL players [0..3]
    for (int p = 0; p < 4; ++p) {
      std::vector<float> obs_direct = d_bf16->GetConsumedObservation(*state, p);
      std::vector<float> obs_batched1 = b1_bf16->GetConsumedObservation(*state, p);
      std::vector<float> obs_batched16 = b16_bf16->GetConsumedObservation(*state, p);

      if (obs_direct.size() != dune_imperium::kFullPublicInformationStateSize) {
        std::cerr << "FAIL: Obs size mismatch at pos " << manifest_id << " player " << p
                  << ": got " << obs_direct.size() << " vs exp "
                  << dune_imperium::kFullPublicInformationStateSize << "\n";
        all_mechanical_checks_pass = false;
      }
      if (obs_direct != obs_batched1 || obs_direct != obs_batched16) {
        std::cerr << "FAIL: Consumed observation binary mismatch at pos " << manifest_id
                  << " player " << p << "\n";
        all_consumed_obs_match = false;
        all_mechanical_checks_pass = false;
      }
    }

    // Candidate Descriptors
    dune_semantic::ExtractCandidateDescriptors(
        *dune, actual_legals, &cand_data_list[i], d_bf16->SemanticDescriptorSchema());
    if (cand_data_list[i].actions != actual_legals) {
      std::cerr << "FAIL: Candidate descriptor actions mismatch at pos " << manifest_id << "\n";
      all_mechanical_checks_pass = false;
    }

    obs_vectors[i] = d_bf16->GetConsumedObservation(*state, expected_players[i]);
    states[i] = std::move(state);
  }

  // --- Start GPU Measurement Window ---
  auto t_start_gpu = std::chrono::steady_clock::now();

  std::vector<std::map<Action, double>> d_bf16_priors(N_POS);
  std::vector<std::map<Action, double>> d_fp32_priors(N_POS);
  std::vector<std::vector<double>> d_bf16_values(N_POS);
  std::vector<std::vector<double>> d_fp32_values(N_POS);

  for (size_t i = 0; i < N_POS; ++i) {
    // Priors
    for (const auto& ap : d_bf16->Prior(*states[i])) d_bf16_priors[i][ap.first] = ap.second;
    for (const auto& ap : d_fp32->Prior(*states[i])) d_fp32_priors[i][ap.first] = ap.second;
    // Values
    d_bf16_values[i] = d_bf16->Evaluate(*states[i]);
    d_fp32_values[i] = d_fp32->Evaluate(*states[i]);
  }

  // 2. Batched Evaluators at MATCHED GEOMETRY (Batch Size 1 for prior, 4 for eval: single-threaded sequential execution)
  std::vector<std::map<Action, double>> b1_bf16_priors(N_POS);
  std::vector<std::map<Action, double>> b1_fp32_priors(N_POS);
  std::vector<std::vector<double>> b1_bf16_values(N_POS);
  std::vector<std::vector<double>> b1_fp32_values(N_POS);

  for (size_t i = 0; i < N_POS; ++i) {
    for (const auto& ap : b1_bf16->Prior(*states[i])) b1_bf16_priors[i][ap.first] = ap.second;
    for (const auto& ap : b1_fp32->Prior(*states[i])) b1_fp32_priors[i][ap.first] = ap.second;
    b1_bf16_values[i] = b1_bf16->Evaluate(*states[i]);
    b1_fp32_values[i] = b1_fp32->Evaluate(*states[i]);
  }

  // 3. Batched Evaluators with 16 Concurrent Threads (VARYING GEOMETRY: physical batch size 16!)
  std::vector<std::map<Action, double>> b16_bf16_priors(N_POS);
  std::vector<std::map<Action, double>> b16_fp32_priors(N_POS);
  std::vector<std::vector<double>> b16_bf16_values(N_POS);
  std::vector<std::vector<double>> b16_fp32_values(N_POS);
  std::vector<open_spiel::CompactEvalResult> b16_bf16_details(N_POS);
  std::vector<open_spiel::CompactEvalResult> b16_fp32_details(N_POS);

  // Evaluate priors concurrently across 16 threads
  {
    SimpleBarrier barrier(16);
    std::vector<std::thread> workers;
    workers.reserve(16);
    for (size_t t = 0; t < 16; ++t) {
      workers.emplace_back([&, t]() {
        barrier.wait(); // All 16 threads hit the barrier and submit simultaneously!
        open_spiel::CompactEvalResult res = b16_bf16->PriorWithDetails(*states[t]);
        b16_bf16_details[t] = res;
        for (size_t k = 0; k < res.actions.size(); ++k) {
          b16_bf16_priors[t][res.actions[k]] = res.probabilities[k];
        }
      });
    }
    for (auto& w : workers) w.join();
  }

  // Evaluate priors concurrently in FP32
  {
    SimpleBarrier barrier(16);
    std::vector<std::thread> workers;
    workers.reserve(16);
    for (size_t t = 0; t < 16; ++t) {
      workers.emplace_back([&, t]() {
        barrier.wait();
        open_spiel::CompactEvalResult res = b16_fp32->PriorWithDetails(*states[t]);
        b16_fp32_details[t] = res;
        for (size_t k = 0; k < res.actions.size(); ++k) {
          b16_fp32_priors[t][res.actions[k]] = res.probabilities[k];
        }
      });
    }
    for (auto& w : workers) w.join();
  }

  // Verify genuine coordinator batch membership
  bool coordinator_membership_verified = true;
  if (b16_bf16_details.size() != 16) {
    std::cerr << "FAIL: Expected 16 batch details, got " << b16_bf16_details.size() << "\n";
    coordinator_membership_verified = false;
  } else {
    int64_t target_batch_id = b16_bf16_details[0].physical_batch_id;
    if (target_batch_id < 0) {
      std::cerr << "FAIL: Thread 0 physical_batch_id invalid: " << target_batch_id << "\n";
      coordinator_membership_verified = false;
    }
    std::set<int32_t> seen_rows;
    for (size_t t = 0; t < 16; ++t) {
      const auto& res = b16_bf16_details[t];
      if (res.physical_batch_id != target_batch_id) {
        std::cerr << "FAIL: Thread " << t << " batch id mismatch: got "
                  << res.physical_batch_id << " vs exp " << target_batch_id << "\n";
        coordinator_membership_verified = false;
      }
      if (res.physical_batch_size != 16) {
        std::cerr << "FAIL: Thread " << t << " batch size mismatch: got "
                  << res.physical_batch_size << " vs exp 16\n";
        coordinator_membership_verified = false;
      }
      if (res.physical_batch_row < 0 || res.physical_batch_row >= 16) {
        std::cerr << "FAIL: Thread " << t << " batch row out of range: "
                  << res.physical_batch_row << "\n";
        coordinator_membership_verified = false;
      }
      if (seen_rows.count(res.physical_batch_row) > 0) {
        std::cerr << "FAIL: Thread " << t << " duplicate batch row: "
                  << res.physical_batch_row << "\n";
        coordinator_membership_verified = false;
      }
      seen_rows.insert(res.physical_batch_row);
    }
    if (seen_rows.size() != 16) {
      std::cerr << "FAIL: b16_bf16 did not observe 16 distinct rows in batch!\n";
      coordinator_membership_verified = false;
    }
  }
  if (!coordinator_membership_verified) {
    all_mechanical_checks_pass = false;
  }

  // Evaluate leaf values concurrently (4 positions * 4 players = 16 rows per batch)
  {
    SimpleBarrier barrier(4);
    std::vector<std::thread> workers;
    workers.reserve(4);
    for (size_t b = 0; b < 4; ++b) {
      workers.emplace_back([&, b]() {
        barrier.wait();
        b16_bf16_values[b] = b16_bf16->Evaluate(*states[b]);
      });
    }
    for (auto& w : workers) w.join();
  }

  {
    SimpleBarrier barrier(4);
    std::vector<std::thread> workers;
    workers.reserve(4);
    for (size_t b = 0; b < 4; ++b) {
      workers.emplace_back([&, b]() {
        barrier.wait();
        b16_fp32_values[b] = b16_fp32->Evaluate(*states[b]);
      });
    }
    for (auto& w : workers) w.join();
  }

  // 3. Detailed Tensor Layer-by-Layer Breakdown (Batch 1 vs Batch 16 in BF16 and FP32)
  // Stack all 16 observations into a single tensor of shape (16, 9182)
  auto tensor_opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  torch::Tensor cpu_stacked = torch::empty({16, model_dim}, tensor_opts_f32);
  float* cpu_stacked_ptr = cpu_stacked.data_ptr<float>();
  for (size_t i = 0; i < 16; ++i) {
    std::memcpy(cpu_stacked_ptr + i * model_dim, obs_vectors[i].data(), model_dim * sizeof(float));
  }
  torch::Tensor dev_stacked = cpu_stacked.to(device);

  struct LayerDiffs {
    double trunk_max_diff;
    double base_logits_max_diff;
    double semantic_corr_max_diff;
    double capped_logits_max_diff;
    double prior_prob_max_diff;
  };

  std::vector<LayerDiffs> bf16_layer_diffs(16);
  std::vector<LayerDiffs> fp32_layer_diffs(16);

  // A. BF16 Layer Breakdown
  {
    torch::InferenceMode guard;
    AutocastGuard autocast_guard(c10::DeviceType::CUDA, true);
    at::globalContext().setAllowTF32CuBLAS(false);

    // Forward full batch (16)
    auto out16 = search_model->forward(dev_stacked);
    torch::Tensor trunk16 = out16.trunk.clone();
    torch::Tensor logits16 = out16.logits.clone();

    std::vector<const dune_semantic::CandidateActionData*> batch_cands(16);
    for (size_t i = 0; i < 16; ++i) batch_cands[i] = &cand_data_list[i];
    dune_semantic::ApplySemanticScorerBatch(
        search_model->semantic_scorer_, trunk16, batch_cands, logits16, device);
    torch::Tensor corrected16 = logits16.clone();

    // Now forward each row individually (batch size 1)
    for (size_t i = 0; i < 16; ++i) {
      torch::Tensor single_obs = dev_stacked.slice(0, i, i + 1);
      auto out1 = search_model->forward(single_obs);
      torch::Tensor trunk1 = out1.trunk;
      torch::Tensor logits1 = out1.logits.clone();

      std::vector<const dune_semantic::CandidateActionData*> single_cand = {&cand_data_list[i]};
      dune_semantic::ApplySemanticScorerBatch(
          search_model->semantic_scorer_, trunk1, single_cand, logits1, device);
      torch::Tensor corrected1 = logits1;

      // Layer 1: Trunk diff
      bf16_layer_diffs[i].trunk_max_diff =
          (trunk16[i].to(torch::kFloat32) - trunk1[0].to(torch::kFloat32)).abs().max().item<double>();

      // Layer 2: Base logits diff
      bf16_layer_diffs[i].base_logits_max_diff =
          (out16.logits[i].to(torch::kFloat32) - out1.logits[0].to(torch::kFloat32)).abs().max().item<double>();

      // Layer 3: Semantic corrections diff
      torch::Tensor corr16 = (corrected16[i] - out16.logits[i]).to(torch::kFloat32);
      torch::Tensor corr1 = (corrected1[0] - out1.logits[0]).to(torch::kFloat32);
      bf16_layer_diffs[i].semantic_corr_max_diff =
          (corr16 - corr1).abs().max().item<double>();

      // Layer 4 & 5: Legal Centering & Capping and Softmax Probabilities
      torch::Tensor row16_cpu = corrected16[i].to(torch::kFloat32).to(torch::kCPU).contiguous();
      torch::Tensor row1_cpu = corrected1[0].to(torch::kFloat32).to(torch::kCPU).contiguous();
      std::vector<float> row16(row16_cpu.data_ptr<float>(),
                              row16_cpu.data_ptr<float>() + row16_cpu.numel());
      std::vector<float> row1(row1_cpu.data_ptr<float>(),
                             row1_cpu.data_ptr<float>() + row1_cpu.numel());

      CenterAndCapLegalLogits(row16, expected_legals[i], 10.0f);
      CenterAndCapLegalLogits(row1, expected_legals[i], 10.0f);

      double max_c_diff = 0.0;
      for (Action a : expected_legals[i]) {
        max_c_diff = std::max(max_c_diff, static_cast<double>(std::abs(row16[a] - row1[a])));
      }
      bf16_layer_diffs[i].capped_logits_max_diff = max_c_diff;

      // Probabilities
      double max_l16 = -1e9, max_l1 = -1e9;
      for (Action a : expected_legals[i]) {
        max_l16 = std::max(max_l16, static_cast<double>(row16[a]));
        max_l1 = std::max(max_l1, static_cast<double>(row1[a]));
      }
      double s16 = 0.0, s1 = 0.0;
      for (Action a : expected_legals[i]) {
        s16 += std::exp(row16[a] - max_l16);
        s1 += std::exp(row1[a] - max_l1);
      }
      double max_p_diff = 0.0;
      for (Action a : expected_legals[i]) {
        double p16 = std::exp(row16[a] - max_l16) / s16;
        double p1 = std::exp(row1[a] - max_l1) / s1;
        max_p_diff = std::max(max_p_diff, std::abs(p16 - p1));
      }
      bf16_layer_diffs[i].prior_prob_max_diff = max_p_diff;
    }
  }

  // B. FP32 Layer Breakdown (Diagnostic Control with TF32 disabled)
  {
    torch::InferenceMode guard;
    AutocastGuard autocast_guard(c10::DeviceType::CUDA, false); // FP32!
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);

    auto out16 = search_model->forward(dev_stacked);
    torch::Tensor trunk16 = out16.trunk.clone();
    torch::Tensor logits16 = out16.logits.clone();

    std::vector<const dune_semantic::CandidateActionData*> batch_cands(16);
    for (size_t i = 0; i < 16; ++i) batch_cands[i] = &cand_data_list[i];
    dune_semantic::ApplySemanticScorerBatch(
        search_model->semantic_scorer_, trunk16, batch_cands, logits16, device);
    torch::Tensor corrected16 = logits16.clone();

    for (size_t i = 0; i < 16; ++i) {
      torch::Tensor single_obs = dev_stacked.slice(0, i, i + 1);
      auto out1 = search_model->forward(single_obs);
      torch::Tensor trunk1 = out1.trunk;
      torch::Tensor logits1 = out1.logits.clone();

      std::vector<const dune_semantic::CandidateActionData*> single_cand = {&cand_data_list[i]};
      dune_semantic::ApplySemanticScorerBatch(
          search_model->semantic_scorer_, trunk1, single_cand, logits1, device);
      torch::Tensor corrected1 = logits1;

      fp32_layer_diffs[i].trunk_max_diff =
          (trunk16[i] - trunk1[0]).abs().max().item<double>();
      fp32_layer_diffs[i].base_logits_max_diff =
          (out16.logits[i] - out1.logits[0]).abs().max().item<double>();

      torch::Tensor corr16 = corrected16[i] - out16.logits[i];
      torch::Tensor corr1 = corrected1[0] - out1.logits[0];
      fp32_layer_diffs[i].semantic_corr_max_diff =
          (corr16 - corr1).abs().max().item<double>();

      torch::Tensor row16_cpu = corrected16[i].to(torch::kFloat32).to(torch::kCPU).contiguous();
      torch::Tensor row1_cpu = corrected1[0].to(torch::kFloat32).to(torch::kCPU).contiguous();
      std::vector<float> row16(row16_cpu.data_ptr<float>(),
                              row16_cpu.data_ptr<float>() + row16_cpu.numel());
      std::vector<float> row1(row1_cpu.data_ptr<float>(),
                             row1_cpu.data_ptr<float>() + row1_cpu.numel());

      CenterAndCapLegalLogits(row16, expected_legals[i], 10.0f);
      CenterAndCapLegalLogits(row1, expected_legals[i], 10.0f);

      double max_c_diff = 0.0;
      for (Action a : expected_legals[i]) {
        max_c_diff = std::max(max_c_diff, static_cast<double>(std::abs(row16[a] - row1[a])));
      }
      fp32_layer_diffs[i].capped_logits_max_diff = max_c_diff;

      double max_l16 = -1e9, max_l1 = -1e9;
      for (Action a : expected_legals[i]) {
        max_l16 = std::max(max_l16, static_cast<double>(row16[a]));
        max_l1 = std::max(max_l1, static_cast<double>(row1[a]));
      }
      double s16 = 0.0, s1 = 0.0;
      for (Action a : expected_legals[i]) {
        s16 += std::exp(row16[a] - max_l16);
        s1 += std::exp(row1[a] - max_l1);
      }
      double max_p_diff = 0.0;
      for (Action a : expected_legals[i]) {
        double p16 = std::exp(row16[a] - max_l16) / s16;
        double p1 = std::exp(row1[a] - max_l1) / s1;
        max_p_diff = std::max(max_p_diff, std::abs(p16 - p1));
      }
      fp32_layer_diffs[i].prior_prob_max_diff = max_p_diff;
    }
  }

  auto t_end_gpu = std::chrono::steady_clock::now();
  double gpu_wall_s = std::chrono::duration<double>(t_end_gpu - t_start_gpu).count();

  // --- Aggregate & Compile Results ---
  json::Object root_report;
  json::Array pos_arr;

  // 1. Matched Geometry Comparison (Direct vs Batched @ Batch Size 1)
  double max_matched_geom_prior_diff_bf16 = 0.0;
  double max_matched_geom_prior_diff_fp32 = 0.0;
  double max_matched_geom_val_diff_bf16 = 0.0;
  double max_matched_geom_val_diff_fp32 = 0.0;

  // 2. Varied Geometry Comparison (Batched @ Batch Size 1 vs Batched @ Batch Size 16)
  double max_varied_geom_prior_diff_bf16 = 0.0;
  double max_varied_geom_prior_diff_fp32 = 0.0;
  double max_varied_geom_val_diff_bf16 = 0.0;
  double max_varied_geom_val_diff_fp32 = 0.0;

  // 3. End-to-End Comparison (Direct Batch 1 vs Batched Batch 16)
  double max_direct_vs_b16_prior_diff_bf16 = 0.0;
  double max_direct_vs_b16_prior_diff_fp32 = 0.0;
  double max_direct_vs_b16_val_diff_bf16 = 0.0;
  double max_direct_vs_b16_val_diff_fp32 = 0.0;

  for (size_t i = 0; i < N_POS; ++i) {
    json::Object p_rec;
    p_rec["manifest_id"] = static_cast<int64_t>(i);
    p_rec["category"] = categories[i];
    p_rec["player"] = static_cast<int64_t>(expected_players[i]);

    // Matched geometry diffs (d vs b1)
    double p_diff_matched_bf16 = 0.0;
    double p_diff_matched_fp32 = 0.0;
    for (Action a : expected_legals[i]) {
      p_diff_matched_bf16 = std::max(p_diff_matched_bf16, std::abs(d_bf16_priors[i][a] - b1_bf16_priors[i][a]));
      p_diff_matched_fp32 = std::max(p_diff_matched_fp32, std::abs(d_fp32_priors[i][a] - b1_fp32_priors[i][a]));
    }
    double v_diff_matched_bf16 = 0.0;
    double v_diff_matched_fp32 = 0.0;
    if (i < 4) {
      for (int p = 0; p < 4; ++p) {
        v_diff_matched_bf16 = std::max(v_diff_matched_bf16, std::abs(d_bf16_values[i][p] - b1_bf16_values[i][p]));
        v_diff_matched_fp32 = std::max(v_diff_matched_fp32, std::abs(d_fp32_values[i][p] - b1_fp32_values[i][p]));
      }
    }

    // Varied geometry diffs (b1 vs b16)
    double p_diff_varied_bf16 = 0.0;
    double p_diff_varied_fp32 = 0.0;
    for (Action a : expected_legals[i]) {
      p_diff_varied_bf16 = std::max(p_diff_varied_bf16, std::abs(b1_bf16_priors[i][a] - b16_bf16_priors[i][a]));
      p_diff_varied_fp32 = std::max(p_diff_varied_fp32, std::abs(b1_fp32_priors[i][a] - b16_fp32_priors[i][a]));
    }
    double v_diff_varied_bf16 = 0.0;
    double v_diff_varied_fp32 = 0.0;
    if (i < 4) {
      for (int p = 0; p < 4; ++p) {
        v_diff_varied_bf16 = std::max(v_diff_varied_bf16, std::abs(b1_bf16_values[i][p] - b16_bf16_values[i][p]));
        v_diff_varied_fp32 = std::max(v_diff_varied_fp32, std::abs(b1_fp32_values[i][p] - b16_fp32_values[i][p]));
      }
    }

    // End-to-end diffs (d vs b16)
    double p_diff_e2e_bf16 = 0.0;
    double p_diff_e2e_fp32 = 0.0;
    for (Action a : expected_legals[i]) {
      p_diff_e2e_bf16 = std::max(p_diff_e2e_bf16, std::abs(d_bf16_priors[i][a] - b16_bf16_priors[i][a]));
      p_diff_e2e_fp32 = std::max(p_diff_e2e_fp32, std::abs(d_fp32_priors[i][a] - b16_fp32_priors[i][a]));
    }
    double v_diff_e2e_bf16 = 0.0;
    double v_diff_e2e_fp32 = 0.0;
    if (i < 4) {
      for (int p = 0; p < 4; ++p) {
        v_diff_e2e_bf16 = std::max(v_diff_e2e_bf16, std::abs(d_bf16_values[i][p] - b16_bf16_values[i][p]));
        v_diff_e2e_fp32 = std::max(v_diff_e2e_fp32, std::abs(d_fp32_values[i][p] - b16_fp32_values[i][p]));
      }
    }

    max_matched_geom_prior_diff_bf16 = std::max(max_matched_geom_prior_diff_bf16, p_diff_matched_bf16);
    max_matched_geom_prior_diff_fp32 = std::max(max_matched_geom_prior_diff_fp32, p_diff_matched_fp32);
    max_matched_geom_val_diff_bf16 = std::max(max_matched_geom_val_diff_bf16, v_diff_matched_bf16);
    max_matched_geom_val_diff_fp32 = std::max(max_matched_geom_val_diff_fp32, v_diff_matched_fp32);

    max_varied_geom_prior_diff_bf16 = std::max(max_varied_geom_prior_diff_bf16, p_diff_varied_bf16);
    max_varied_geom_prior_diff_fp32 = std::max(max_varied_geom_prior_diff_fp32, p_diff_varied_fp32);
    max_varied_geom_val_diff_bf16 = std::max(max_varied_geom_val_diff_bf16, v_diff_varied_bf16);
    max_varied_geom_val_diff_fp32 = std::max(max_varied_geom_val_diff_fp32, v_diff_varied_fp32);

    max_direct_vs_b16_prior_diff_bf16 = std::max(max_direct_vs_b16_prior_diff_bf16, p_diff_e2e_bf16);
    max_direct_vs_b16_prior_diff_fp32 = std::max(max_direct_vs_b16_prior_diff_fp32, p_diff_e2e_fp32);
    max_direct_vs_b16_val_diff_bf16 = std::max(max_direct_vs_b16_val_diff_bf16, v_diff_e2e_bf16);
    max_direct_vs_b16_val_diff_fp32 = std::max(max_direct_vs_b16_val_diff_fp32, v_diff_e2e_fp32);

    p_rec["matched_geom_prior_diff_bf16"] = p_diff_matched_bf16;
    p_rec["matched_geom_prior_diff_fp32"] = p_diff_matched_fp32;
    p_rec["matched_geom_val_diff_bf16"] = v_diff_matched_bf16;
    p_rec["matched_geom_val_diff_fp32"] = v_diff_matched_fp32;

    p_rec["varied_geom_prior_diff_bf16"] = p_diff_varied_bf16;
    p_rec["varied_geom_prior_diff_fp32"] = p_diff_varied_fp32;
    p_rec["varied_geom_val_diff_bf16"] = v_diff_varied_bf16;
    p_rec["varied_geom_val_diff_fp32"] = v_diff_varied_fp32;

    p_rec["prior_diff_bf16_batch16_vs_direct"] = p_diff_e2e_bf16;
    p_rec["prior_diff_fp32_batch16_vs_direct"] = p_diff_e2e_fp32;
    p_rec["val_diff_bf16"] = v_diff_e2e_bf16;
    p_rec["val_diff_fp32"] = v_diff_e2e_fp32;

    // Layer breakdown
    json::Object bf16_layer;
    bf16_layer["trunk_max_diff"] = bf16_layer_diffs[i].trunk_max_diff;
    bf16_layer["base_logits_max_diff"] = bf16_layer_diffs[i].base_logits_max_diff;
    bf16_layer["semantic_corr_max_diff"] = bf16_layer_diffs[i].semantic_corr_max_diff;
    bf16_layer["capped_logits_max_diff"] = bf16_layer_diffs[i].capped_logits_max_diff;
    bf16_layer["prior_prob_max_diff"] = bf16_layer_diffs[i].prior_prob_max_diff;
    p_rec["bf16_layer_breakdown"] = bf16_layer;

    json::Object fp32_layer;
    fp32_layer["trunk_max_diff"] = fp32_layer_diffs[i].trunk_max_diff;
    fp32_layer["base_logits_max_diff"] = fp32_layer_diffs[i].base_logits_max_diff;
    fp32_layer["semantic_corr_max_diff"] = fp32_layer_diffs[i].semantic_corr_max_diff;
    fp32_layer["capped_logits_max_diff"] = fp32_layer_diffs[i].capped_logits_max_diff;
    fp32_layer["prior_prob_max_diff"] = fp32_layer_diffs[i].prior_prob_max_diff;
    p_rec["fp32_layer_breakdown"] = fp32_layer;

    pos_arr.push_back(p_rec);

    std::cout << "Pos " << std::setw(2) << i << " (" << std::setw(18) << categories[i] << "): "
              << "Matched(D vs B1 BF16)=" << std::scientific << std::setprecision(2) << p_diff_matched_bf16
              << " | Varied(B1 vs B16 BF16)=" << p_diff_varied_bf16
              << " | Varied(B1 vs B16 FP32)=" << p_diff_varied_fp32
              << " | TrunkDiff BF16=" << bf16_layer_diffs[i].trunk_max_diff
              << "\n";
  }

  // Finite output and probability normalization checks
  bool all_finite_and_valid = true;
  for (size_t i = 0; i < N_POS; ++i) {
    auto check_dist = [&](const std::string& lbl, const std::map<Action, double>& priors,
                          const std::vector<double>& values) {
      double sum_p = 0.0;
      for (Action a : expected_legals[i]) {
        auto it = priors.find(a);
        if (it == priors.end()) {
          std::cerr << "FAIL: Missing legal action " << a << " in " << lbl << " pos " << i << "\n";
          all_finite_and_valid = false;
          continue;
        }
        double p = it->second;
        if (!std::isfinite(p) || p < 0.0) {
          std::cerr << "FAIL: Invalid prob " << p << " in " << lbl << " pos " << i << " action " << a << "\n";
          all_finite_and_valid = false;
        }
        sum_p += p;
      }
      if (std::abs(sum_p - 1.0) > 1e-5) {
        std::cerr << "FAIL: Prob sum " << sum_p << " != 1.0 in " << lbl << " pos " << i << "\n";
        all_finite_and_valid = false;
      }
      for (size_t p = 0; p < values.size(); ++p) {
        if (!std::isfinite(values[p])) {
          std::cerr << "FAIL: Non-finite value in " << lbl << " pos " << i << " player " << p << "\n";
          all_finite_and_valid = false;
        }
      }
    };

    check_dist("d_bf16", d_bf16_priors[i], d_bf16_values[i]);
    check_dist("d_fp32", d_fp32_priors[i], d_fp32_values[i]);
    check_dist("b1_bf16", b1_bf16_priors[i], b1_bf16_values[i]);
    check_dist("b1_fp32", b1_fp32_priors[i], b1_fp32_values[i]);
    check_dist("b16_bf16", b16_bf16_priors[i], b16_bf16_values[i]);
    check_dist("b16_fp32", b16_fp32_priors[i], b16_fp32_values[i]);
  }

  // Physical batch row mapping from genuine coordinator captures
  json::Array batch_mapping;
  for (size_t i = 0; i < N_POS; ++i) {
    json::Object m;
    m["batch_id"] = static_cast<int64_t>(b16_bf16_details[i].physical_batch_id);
    m["batch_size"] = static_cast<int64_t>(b16_bf16_details[i].physical_batch_size);
    m["batch_row"] = static_cast<int64_t>(b16_bf16_details[i].physical_batch_row);
    m["thread_id"] = static_cast<int64_t>(i);
    m["manifest_id"] = static_cast<int64_t>(i);
    m["player"] = static_cast<int64_t>(expected_players[i]);
    m["category"] = categories[i];
    batch_mapping.push_back(m);
  }

  root_report["positions"] = pos_arr;
  root_report["physical_batch_row_mapping"] = batch_mapping;
  root_report["coordinator_batch_membership_verified"] = coordinator_membership_verified;
  root_report["consumed_observations_exact_match"] = all_consumed_obs_match;
  root_report["finite_and_valid_outputs_pass"] = all_finite_and_valid;
  root_report["gpu_wall_time_s"] = gpu_wall_s;

  root_report["max_matched_geom_prior_diff_bf16"] = max_matched_geom_prior_diff_bf16;
  root_report["max_matched_geom_prior_diff_fp32"] = max_matched_geom_prior_diff_fp32;
  root_report["max_matched_geom_val_diff_bf16"] = max_matched_geom_val_diff_bf16;
  root_report["max_matched_geom_val_diff_fp32"] = max_matched_geom_val_diff_fp32;

  root_report["max_varied_geom_prior_diff_bf16"] = max_varied_geom_prior_diff_bf16;
  root_report["max_varied_geom_prior_diff_fp32"] = max_varied_geom_prior_diff_fp32;
  root_report["max_varied_geom_val_diff_bf16"] = max_varied_geom_val_diff_bf16;
  root_report["max_varied_geom_val_diff_fp32"] = max_varied_geom_val_diff_fp32;

  root_report["max_prior_diff_bf16_batch16_vs_direct"] = max_direct_vs_b16_prior_diff_bf16;
  root_report["max_prior_diff_fp32_batch16_vs_direct"] = max_direct_vs_b16_prior_diff_fp32;
  root_report["max_val_diff_bf16"] = max_direct_vs_b16_val_diff_bf16;
  root_report["max_val_diff_fp32"] = max_direct_vs_b16_val_diff_fp32;

  // Decision Rules
  const bool matched_geometry_equivalence_pass =
      (max_matched_geom_prior_diff_bf16 <= 1e-6) &&
      (max_matched_geom_prior_diff_fp32 <= 1e-6) &&
      (max_matched_geom_val_diff_bf16 <= 1e-6) &&
      (max_matched_geom_val_diff_fp32 <= 1e-6);

  const bool fp32_controls_pass =
      (max_varied_geom_prior_diff_fp32 <= 1e-4) &&
      (max_varied_geom_val_diff_fp32 <= 1e-4);

  const bool bf16_divergence_reproduced_by_batch_geometry =
      (max_varied_geom_prior_diff_bf16 > 1e-4) &&
      (max_varied_geom_prior_diff_bf16 <= 0.05);

  root_report["all_mechanical_checks_pass"] = all_mechanical_checks_pass;
  root_report["matched_geometry_equivalence_pass"] = matched_geometry_equivalence_pass;
  root_report["fp32_controls_pass"] = fp32_controls_pass;
  root_report["bf16_divergence_reproduced_by_batch_geometry"] = bf16_divergence_reproduced_by_batch_geometry;
  root_report["time_cap_exceeded"] = (gpu_wall_s > 600.0);

  std::string verdict;
  if (!all_mechanical_checks_pass || !all_consumed_obs_match || !all_finite_and_valid) {
    verdict = "FAIL_MECHANICAL_CHECKS";
  } else if (!matched_geometry_equivalence_pass) {
    verdict = "FAIL_EVALUATOR_IMPLEMENTATION_MISMATCH";
  } else if (!fp32_controls_pass) {
    verdict = "FAIL_FP32_CONTROL";
  } else if (bf16_divergence_reproduced_by_batch_geometry) {
    verdict = "PASS_WITH_MEASURED_DRIFT";
  } else if (max_varied_geom_prior_diff_bf16 <= 1e-4) {
    verdict = "PASS_EXACT";
  } else {
    verdict = "FAIL_UNEXPECTED_DRIFT";
  }
  root_report["verdict"] = verdict;

  std::cout << "\n======================================================\n";
  std::cout << "Stage A Diagnosis Verdict:            " << verdict << "\n";
  std::cout << "Mechanical Checks:                      " << (all_mechanical_checks_pass ? "PASS" : "FAIL") << "\n";
  std::cout << "Matched Geometry Equivalence (<= 1e-6): " << (matched_geometry_equivalence_pass ? "PASS" : "FAIL") << "\n";
  std::cout << "  - Prior Diff BF16 Matched:            " << std::scientific << max_matched_geom_prior_diff_bf16 << "\n";
  std::cout << "  - Prior Diff FP32 Matched:            " << std::scientific << max_matched_geom_prior_diff_fp32 << "\n";
  std::cout << "FP32 Controls (<= 1e-4):                " << (fp32_controls_pass ? "PASS" : "FAIL") << "\n";
  std::cout << "  - Varied Geom Prior Diff FP32:        " << std::scientific << max_varied_geom_prior_diff_fp32 << "\n";
  std::cout << "  - Varied Geom Val Diff FP32:          " << std::scientific << max_varied_geom_val_diff_fp32 << "\n";
  std::cout << "BF16 Geometry Drift:                    \n";
  std::cout << "  - Varied Geom Prior Diff BF16:        " << std::scientific << max_varied_geom_prior_diff_bf16 << "\n";
  std::cout << "  - Direct vs Batch16 Prior Diff BF16:  " << std::scientific << max_direct_vs_b16_prior_diff_bf16 << "\n";
  std::cout << "GPU Wall Clock:                         " << std::fixed << std::setprecision(2) << gpu_wall_s << " s (Cap: 600 s)\n";
  std::cout << "======================================================\n";

  if (!out_json_path.empty()) {
    std::ofstream of(out_json_path);
    if (of.is_open()) {
      of << json::ToString(root_report, /*wrap=*/true) << "\n";
      std::cout << "Wrote detailed Stage A JSON report to: " << out_json_path << "\n";
    }
  }

  return (verdict == "FAIL_MECHANICAL_CHECKS" ||
          verdict == "FAIL_EVALUATOR_IMPLEMENTATION_MISMATCH" ||
          verdict == "FAIL_FP32_CONTROL" ||
          verdict == "FAIL_UNEXPECTED_DRIFT") ? 1 : 0;
}

} // namespace open_spiel

int main(int argc, char* argv[]) {
  return open_spiel::RunProbe(argc, argv);
}
