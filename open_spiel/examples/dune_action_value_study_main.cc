#include "dune_action_value_study.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"

ABSL_FLAG(std::string, phase, "all", "Execution phase: all, verify, pilot, corpus, rollout_train_dev, train, rollout_test, evaluate");
ABSL_FLAG(std::string, output_dir, "", "Output directory on Storage");
ABSL_FLAG(std::string, model_path, open_spiel::action_value_study::kActorModelPathDefault, "Frozen actor model checkpoint");
ABSL_FLAG(int, threads, 16, "Number of worker threads");
ABSL_FLAG(int, eval_batch_size, 64, "BatchedEvaluator target batch size");
ABSL_FLAG(int, eval_timeout_ms, 2, "BatchedEvaluator timeout ms");

namespace open_spiel {
namespace action_value_study {

// Forward declarations of test suite from dune_action_value_study_test.cc
void TestSchemaValidation();
void TestActorRelativeMapping();
void TestSeedDerivationAndPairing();
void TestCandidateActionSelection();
void TestCriticOptimizerStepAndSaveReload();
void TestBootstrapConfidenceIntervals();
void TestManifestContractAndCorruptionRejection();

// Global shared evaluator for actor
std::shared_ptr<SharedDunePolicyValueNetImpl> g_actor_model = nullptr;
std::shared_ptr<BatchedEvaluator> g_actor_evaluator = nullptr;
std::shared_mutex g_sync_mutex;
std::string g_actor_sha256;

void InitializeActor(const std::string& path, const torch::Device& device, int batch_size, int timeout_ms) {
  std::cout << "Loading frozen actor from: " << path << std::endl;
  g_actor_sha256 = ComputeFileSHA256(path);
  std::cout << "Actor SHA256: " << g_actor_sha256 << std::endl;
  SPIEL_CHECK_EQ(g_actor_sha256, kExpectedActorSha256);

  g_actor_model = std::make_shared<SharedDunePolicyValueNetImpl>(
      kActorInputDim, kActorHiddenDim, kActionDim, kActorNumBlocks, false);
  torch::serialize::InputArchive archive;
  archive.load_from(path, device);
  g_actor_model->load(archive);
  g_actor_model->to(device);
  g_actor_model->eval();

  // Pure FP32, TF32 disabled
  at::globalContext().setAllowTF32CuBLAS(false);
  at::globalContext().setAllowTF32CuDNN(false);

  g_actor_evaluator = std::make_shared<BatchedEvaluator>(
      g_actor_model, batch_size, timeout_ms, device, &g_sync_mutex,
      0.0f, /*device_synchronize=*/false, /*high_priority_stream=*/false,
      /*emit_batch_membership=*/false, /*rollout_amp=*/false, /*allow_tf32=*/false);
}

// Rollout an episode from a given state using separate chance and policy RNG streams
std::array<double, kNumPlayers> RunRollout(
    const std::shared_ptr<const Game>& game,
    std::unique_ptr<State> state,
    uint64_t chance_seed,
    uint64_t policy_seed) {
  std::mt19937_64 chance_rng(chance_seed);
  std::mt19937_64 policy_rng(policy_seed);

  int step_count = 0;
  while (!state->IsTerminal()) {
    ++step_count;
    if (step_count > 5000) {
      SpielFatalError("RunRollout exceeded 5000 steps");
    }
    if (state->IsChanceNode()) {
      auto outcomes = state->ChanceOutcomes();
      Action action = SampleAction(outcomes, chance_rng).first;
      state->ApplyAction(action);
    } else if (state->CurrentPlayer() == kSimultaneousPlayerId) {
      std::vector<Action> joint;
      for (int p = 0; p < state->NumPlayers(); ++p) {
        auto acts = state->LegalActions(p);
        std::uniform_int_distribution<size_t> d(0, acts.size() - 1);
        joint.push_back(acts[d(chance_rng)]);
      }
      state->ApplyActions(joint);
    } else {
      Player current_player = state->CurrentPlayer();
      std::vector<Action> legal = state->LegalActions();
      SPIEL_CHECK_FALSE(legal.empty());

      std::vector<float> obs(kActorInputDim, 0.0f);
      state->InformationStateTensor(current_player, absl::MakeSpan(obs));

      EvalResult eval_res = g_actor_evaluator->Evaluate(obs);
      std::vector<float> logits = std::move(eval_res.logits);
      CenterAndCapLegalLogits(logits, legal, 10.0f);

      auto sampled = SamplePolicyDistribution(&policy_rng, logits, legal, nullptr);
      state->ApplyAction(sampled.action);
    }
  }

  std::vector<double> ret = state->Returns();
  SPIEL_CHECK_EQ(ret.size(), kNumPlayers);
  std::array<double, kNumPlayers> result{};
  for (int i = 0; i < kNumPlayers; ++i) {
    result[i] = ret[i];
  }
  return result;
}

// Phase: verify
void RunVerify() {
  std::cout << "\n=======================================================\n";
  std::cout << "PHASE: Verification and Corruption Rejection\n";
  std::cout << "=======================================================\n";
  TestSchemaValidation();
  TestActorRelativeMapping();
  TestSeedDerivationAndPairing();
  TestCandidateActionSelection();
  TestCriticOptimizerStepAndSaveReload();
  TestBootstrapConfidenceIntervals();
  TestManifestContractAndCorruptionRejection();

  // Check frozen actor SHA256 matches specification
  std::string sha = ComputeFileSHA256(absl::GetFlag(FLAGS_model_path));
  std::cout << "Verified actor model sha256: " << sha << std::endl;
  SPIEL_CHECK_EQ(sha, kExpectedActorSha256);

  std::cout << "ALL VERIFICATION CHECKS PASSED!\n";
}

// Phase: pilot
void RunPilot(const std::shared_ptr<const Game>& game, const std::filesystem::path& out_dir, int num_threads) {
  std::cout << "\n=======================================================\n";
  std::cout << "PHASE: Disposable 4-Root, 8-Continuation Pilot\n";
  std::cout << "=======================================================\n";

  auto start_time = std::chrono::steady_clock::now();
  std::vector<RootRecord> pilot_roots;

  // Generate 4 natural source games with u15828 to find 4 distinct roots
  for (int ep = 0; ep < 4; ++ep) {
    uint64_t game_seed = DeriveSourceGameSeed(Partition::kPilot, ep);
    std::mt19937_64 chance_rng(game_seed);
    std::mt19937_64 policy_rng(dune_seed::DeriveSeed(kDomainPilot, kStreamContinuationPolicy, game_seed));

    auto state = game->NewInitialState();
    bool selected = false;
    RootRecord rec;

    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        auto outcomes = state->ChanceOutcomes();
        state->ApplyAction(SampleAction(outcomes, chance_rng).first);
        continue;
      }
      if (state->CurrentPlayer() == kSimultaneousPlayerId) {
        std::vector<Action> joint;
        for (int p = 0; p < state->NumPlayers(); ++p) {
          auto acts = state->LegalActions(p);
          std::uniform_int_distribution<size_t> d(0, acts.size() - 1);
          joint.push_back(acts[d(chance_rng)]);
        }
        state->ApplyActions(joint);
        continue;
      }

      Player player = state->CurrentPlayer();
      auto legal = state->LegalActions();
      DuneDecisionRole role = ClassifyDuneDecisionRole(*state, player, false);
      const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
      int current_round = dune_state ? dune_state->GetCurrentRound() : 1;

      std::vector<float> obs(kActorInputDim, 0.0f);
      state->InformationStateTensor(player, absl::MakeSpan(obs));
      EvalResult eval_res = g_actor_evaluator->Evaluate(obs);
      std::vector<float> logits = std::move(eval_res.logits);

      if (!selected && role == DuneDecisionRole::kAgentPrimary && legal.size() >= 2 && current_round >= 2) {
        rec.partition = Partition::kPilot;
        rec.source_episode_id = ep;
        rec.acting_player = player;
        rec.round = current_round;
        rec.stratum = "agent_primary";
        rec.role = role;
        rec.history = state->History();
        rec.legal_actions = legal;
        rec.root_id = pwo2::HistoryHash(rec.history).substr(0, 16);
        rec.candidate_actions = SelectCandidateActions(legal, logits, Partition::kPilot, rec.root_id, &rec.candidate_actor_probs);
        rec.reference_action = rec.candidate_actions[0];
        rec.critic_input_9647 = ExtractCriticInput(*state, player);
        selected = true;
      }

      CenterAndCapLegalLogits(logits, legal, 10.0f);
      auto sampled = SamplePolicyDistribution(&policy_rng, logits, legal, nullptr);
      state->ApplyAction(sampled.action);
    }
    SPIEL_CHECK_TRUE(selected);
    pilot_roots.push_back(std::move(rec));
  }

  std::cout << "Selected 4 pilot roots. Running 8 continuations per candidate action...\n";
  std::atomic<int> completed_continuations{0};
  int total_pilot_continuations = 0;
  for (const auto& r : pilot_roots) {
    total_pilot_continuations += r.candidate_actions.size() * 8;
  }

  struct PilotTask {
    const RootRecord* root;
    Action action;
    int replicate;
  };
  std::vector<PilotTask> tasks;
  for (const auto& r : pilot_roots) {
    for (Action a : r.candidate_actions) {
      for (int k = 0; k < 8; ++k) {
        tasks.push_back({&r, a, k});
      }
    }
  }

  auto rollout_start = std::chrono::steady_clock::now();
  std::atomic<size_t> next_task{0};
  std::vector<std::thread> workers;
  for (int tid = 0; tid < num_threads; ++tid) {
    workers.emplace_back([&]() {
      while (true) {
        size_t idx = next_task.fetch_add(1);
        if (idx >= tasks.size()) break;
        const auto& t = tasks[idx];
        auto state = ReconstructState(game, t.root->history);
        state->ApplyAction(t.action);
        uint64_t c_seed = DeriveContinuationChanceSeed(Partition::kPilot, t.root->root_id, t.replicate);
        uint64_t p_seed = DeriveContinuationPolicySeed(Partition::kPilot, t.root->root_id, t.replicate);
        auto returns = RunRollout(game, std::move(state), c_seed, p_seed);
        completed_continuations.fetch_add(1);
      }
    });
  }
  for (auto& w : workers) {
    w.join();
  }
  auto rollout_end = std::chrono::steady_clock::now();
  double rollout_sec = std::chrono::duration<double>(rollout_end - rollout_start).count();
  double throughput = completed_continuations.load() / rollout_sec;

  // Total continuations for main corpus:
  // Train: 2048 * ~2.9 actions * 16 ~ 95,000
  // Dev: 256 * ~2.9 actions * 32 ~ 23,700
  // Test: 512 * ~2.9 actions * 64 ~ 95,000
  // Total ~ 214,000
  double estimated_total_conts = 214000.0;
  double estimated_total_sec = estimated_total_conts / throughput;
  double deadline_margin_sec = kTotalDeadlineSeconds * 0.85; // 15% margin = 24480s (6.8 hours)

  std::cout << absl::StrFormat("Pilot completed: %d continuations in %.2f s (%.1f conts/sec)\n",
                               completed_continuations.load(), rollout_sec, throughput);
  std::cout << absl::StrFormat("Estimated full corpus rollout time: %.1f s (%.2f hours)\n",
                               estimated_total_sec, estimated_total_sec / 3600.0);
  std::cout << absl::StrFormat("Deadline with 15%% margin: %.1f s (6.8 hours)\n", deadline_margin_sec);

  json::Object report;
  report["pilot_roots"] = static_cast<int64_t>(pilot_roots.size());
  report["pilot_continuations"] = static_cast<int64_t>(completed_continuations.load());
  report["elapsed_sec"] = rollout_sec;
  report["throughput_conts_per_sec"] = throughput;
  report["estimated_total_conts"] = estimated_total_conts;
  report["estimated_total_sec"] = estimated_total_sec;
  report["fits_deadline_margin"] = (estimated_total_sec <= deadline_margin_sec);

  std::ofstream out_f(out_dir / "pilot_throughput_report.json");
  out_f << json::ToString(report) << "\n";
  out_f.close();

  if (estimated_total_sec > deadline_margin_sec) {
    SpielFatalError(absl::StrFormat("Cannot fit 8-hour deadline with 15%% margin (estimated %.1f h > 6.8 h)",
                                    estimated_total_sec / 3600.0));
  }
  std::cout << "PILOT PASSED! Timing margin verified.\n";
}

// Phase: generate_corpus
void RunGenerateCorpus(const std::shared_ptr<const Game>& game, const std::filesystem::path& out_dir, int num_threads) {
  std::cout << "\n=======================================================\n";
  std::cout << "PHASE: Generate Corpus (2,816 Roots)\n";
  std::cout << "=======================================================\n";

  std::vector<RootRecord> all_roots;
  all_roots.reserve(kRootsTotal);

  // Targets per partition and seat
  struct PartitionPlan {
    Partition partition;
    int target_total;
    int target_per_seat;
  };
  std::vector<PartitionPlan> plans = {
      {Partition::kTrain, kRootsTrain, kRootsTrain / 4}, // 2048, 512/seat
      {Partition::kDev,   kRootsDev,   kRootsDev / 4},   // 256, 64/seat
      {Partition::kTest,  kRootsTest,  kRootsTest / 4},  // 512, 128/seat
  };

  int global_episode_counter = 0;
  std::map<int, int> round_distribution;
  std::map<int, int> seat_distribution;

  for (const auto& plan : plans) {
    std::cout << "Generating partition: " << PartitionToString(plan.partition)
              << " (target: " << plan.target_total << " roots, " << plan.target_per_seat << " per seat)...\n";

    std::array<int, 4> seat_counts = {0, 0, 0, 0};
    int partition_collected = 0;

    while (partition_collected < plan.target_total) {
      int ep = global_episode_counter++;
      uint64_t game_seed = DeriveSourceGameSeed(plan.partition, ep);
      std::mt19937_64 chance_rng(game_seed);
      std::mt19937_64 policy_rng(dune_seed::DeriveSeed(PartitionDomain(plan.partition), kStreamContinuationPolicy, game_seed));

      auto state = game->NewInitialState();
      bool found_root = false;
      RootRecord candidate_record;

      // Desired target seat for this game (round-robin among needed seats)
      Player target_seat = -1;
      int min_count = 1e9;
      for (int s = 0; s < 4; ++s) {
        if (seat_counts[s] < plan.target_per_seat && seat_counts[s] < min_count) {
          min_count = seat_counts[s];
          target_seat = s;
        }
      }
      if (target_seat == -1) break;

      while (!state->IsTerminal()) {
        if (state->IsChanceNode()) {
          auto outcomes = state->ChanceOutcomes();
          state->ApplyAction(SampleAction(outcomes, chance_rng).first);
          continue;
        }
        if (state->CurrentPlayer() == kSimultaneousPlayerId) {
          std::vector<Action> joint;
          for (int p = 0; p < state->NumPlayers(); ++p) {
            auto acts = state->LegalActions(p);
            std::uniform_int_distribution<size_t> d(0, acts.size() - 1);
            joint.push_back(acts[d(chance_rng)]);
          }
          state->ApplyActions(joint);
          continue;
        }

        Player player = state->CurrentPlayer();
        auto legal = state->LegalActions();
        DuneDecisionRole role = ClassifyDuneDecisionRole(*state, player, false);
        const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
        int current_round = dune_state ? dune_state->GetCurrentRound() : 1;

        std::vector<float> obs(kActorInputDim, 0.0f);
        state->InformationStateTensor(player, absl::MakeSpan(obs));
        EvalResult eval_res = g_actor_evaluator->Evaluate(obs);
        std::vector<float> logits = std::move(eval_res.logits);

        // Deterministic eligibility check:
        // agent-primary placement, legal >= 2, rounds 2..10, matching target seat
        if (!found_root && role == DuneDecisionRole::kAgentPrimary && legal.size() >= 2 &&
            current_round >= 2 && current_round <= 10 && player == target_seat) {
          candidate_record.partition = plan.partition;
          candidate_record.source_episode_id = ep;
          candidate_record.acting_player = player;
          candidate_record.round = current_round;
          candidate_record.stratum = "agent_primary";
          candidate_record.role = role;
          candidate_record.history = state->History();
          candidate_record.legal_actions = legal;
          candidate_record.root_id = pwo2::HistoryHash(candidate_record.history).substr(0, 16);
          candidate_record.candidate_actions = SelectCandidateActions(
              legal, logits, plan.partition, candidate_record.root_id, &candidate_record.candidate_actor_probs);
          candidate_record.reference_action = candidate_record.candidate_actions[0];
          candidate_record.critic_input_9647 = ExtractCriticInput(*state, player);
          found_root = true;
        }

        CenterAndCapLegalLogits(logits, legal, 10.0f);
        auto sampled = SamplePolicyDistribution(&policy_rng, logits, legal, nullptr);
        state->ApplyAction(sampled.action);
      }

      if (found_root) {
        seat_counts[target_seat]++;
        partition_collected++;
        round_distribution[candidate_record.round]++;
        seat_distribution[candidate_record.acting_player]++;
        all_roots.push_back(std::move(candidate_record));
        if (partition_collected % 256 == 0 || partition_collected == plan.target_total) {
          std::cout << absl::StrFormat("  [%s] Collected %d / %d roots (seats: %d, %d, %d, %d)\n",
                                       PartitionToString(plan.partition), partition_collected,
                                       plan.target_total, seat_counts[0], seat_counts[1],
                                       seat_counts[2], seat_counts[3]);
        }
      }
    }
  }

  SPIEL_CHECK_EQ(all_roots.size(), kRootsTotal);

  // Write corpus manifest and records
  ExecutableManifest manifest;
  std::ofstream manifest_f(out_dir / "corpus_manifest.json");
  manifest_f << manifest.ToJsonString() << "\n";
  manifest_f.close();

  std::ofstream roots_f(out_dir / "corpus_roots.jsonl");
  for (const auto& r : all_roots) {
    json::Object obj;
    obj["root_id"] = r.root_id;
    obj["partition"] = PartitionToString(r.partition);
    obj["source_episode_id"] = static_cast<int64_t>(r.source_episode_id);
    obj["acting_player"] = static_cast<int64_t>(r.acting_player);
    obj["round"] = static_cast<int64_t>(r.round);
    obj["stratum"] = r.stratum;

    json::Array hist;
    for (Action a : r.history) hist.push_back(static_cast<int64_t>(a));
    obj["history"] = hist;

    json::Array leg;
    for (Action a : r.legal_actions) leg.push_back(static_cast<int64_t>(a));
    obj["legal_actions"] = leg;

    json::Array cands;
    for (Action a : r.candidate_actions) cands.push_back(static_cast<int64_t>(a));
    obj["candidate_actions"] = cands;

    obj["reference_action"] = static_cast<int64_t>(r.reference_action);

    json::Array probs;
    for (double p : r.candidate_actor_probs) probs.push_back(p);
    obj["candidate_actor_probs"] = probs;

    json::Array cinp;
    for (float v : r.critic_input_9647) cinp.push_back(static_cast<double>(v));
    obj["critic_input_9647"] = cinp;

    roots_f << json::ToString(obj) << "\n";
  }
  roots_f.close();

  std::cout << "\nCORPUS GENERATION COMPLETE. Total roots: " << all_roots.size() << "\n";
  std::cout << "Round distribution:\n";
  for (const auto& kv : round_distribution) {
    std::cout << absl::StrFormat("  Round %2d: %5d roots (%.1f%%)\n",
                                 kv.first, kv.second, 100.0 * kv.second / all_roots.size());
  }
  std::cout << "Seat distribution:\n";
  for (const auto& kv : seat_distribution) {
    std::cout << absl::StrFormat("  Seat %d: %5d roots (%.1f%%)\n",
                                 kv.first, kv.second, 100.0 * kv.second / all_roots.size());
  }
}

// Load roots from jsonl
std::vector<RootRecord> LoadCorpusRoots(const std::filesystem::path& path) {
  std::ifstream f(path);
  if (!f.is_open()) SpielFatalError("Cannot open roots file: " + path.string());
  std::string line;
  std::vector<RootRecord> roots;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    auto parsed = json::FromString(line);
    SPIEL_CHECK_TRUE(parsed.has_value());
    const auto& obj = parsed.value().GetObject();

    RootRecord r;
    r.root_id = obj.at("root_id").GetString();
    r.partition = StringToPartition(obj.at("partition").GetString());
    r.source_episode_id = static_cast<int>(obj.at("source_episode_id").GetInt());
    r.acting_player = static_cast<Player>(obj.at("acting_player").GetInt());
    r.round = static_cast<int>(obj.at("round").GetInt());
    r.stratum = obj.at("stratum").GetString();
    r.reference_action = static_cast<Action>(obj.at("reference_action").GetInt());

    for (const auto& v : obj.at("history").GetArray()) {
      r.history.push_back(static_cast<Action>(v.GetInt()));
    }
    for (const auto& v : obj.at("legal_actions").GetArray()) {
      r.legal_actions.push_back(static_cast<Action>(v.GetInt()));
    }
    for (const auto& v : obj.at("candidate_actions").GetArray()) {
      r.candidate_actions.push_back(static_cast<Action>(v.GetInt()));
    }
    for (const auto& v : obj.at("candidate_actor_probs").GetArray()) {
      r.candidate_actor_probs.push_back(v.GetDouble());
    }
    for (const auto& v : obj.at("critic_input_9647").GetArray()) {
      r.critic_input_9647.push_back(static_cast<float>(v.GetDouble()));
    }
    roots.push_back(std::move(r));
  }
  return roots;
}

// Phase: rollout_train_dev
void RunRolloutTrainDev(const std::shared_ptr<const Game>& game, const std::filesystem::path& out_dir, int num_threads) {
  std::cout << "\n=======================================================\n";
  std::cout << "PHASE: Rollout Train and Dev Partitions\n";
  std::cout << "=======================================================\n";

  auto roots = LoadCorpusRoots(out_dir / "corpus_roots.jsonl");
  std::filesystem::path conts_path = out_dir / "train_dev_continuations.jsonl";
  std::ofstream out_f(conts_path, std::ios::app);

  std::atomic<int> completed{0};
  int total_targets = 0;
  for (const auto& r : roots) {
    if (r.partition == Partition::kTrain) {
      total_targets += r.candidate_actions.size() * kContinuationsTrain;
    } else if (r.partition == Partition::kDev) {
      total_targets += r.candidate_actions.size() * kContinuationsDev;
    }
  }
  std::cout << "Total train & dev continuations to run: " << total_targets << "\n";

  auto start_time = std::chrono::steady_clock::now();
  std::mutex write_mutex;

  // Parallelize over roots
  std::atomic<size_t> next_root_idx{0};
  std::vector<std::thread> workers;

  for (int tid = 0; tid < num_threads; ++tid) {
    workers.emplace_back([&]() {
      while (true) {
        size_t idx = next_root_idx.fetch_add(1);
        if (idx >= roots.size()) break;
        const auto& r = roots[idx];
        if (r.partition != Partition::kTrain && r.partition != Partition::kDev) continue;

        int num_reps = (r.partition == Partition::kTrain) ? kContinuationsTrain : kContinuationsDev;
        std::vector<ContinuationRecord> batch_records;

        for (Action a : r.candidate_actions) {
          for (int k = 0; k < num_reps; ++k) {
            uint64_t c_seed = DeriveContinuationChanceSeed(r.partition, r.root_id, k);
            uint64_t p_seed = DeriveContinuationPolicySeed(r.partition, r.root_id, k);

            auto state = ReconstructState(game, r.history);
            state->ApplyAction(a);
            auto abs_returns = RunRollout(game, std::move(state), c_seed, p_seed);
            auto rel_returns = ConvertAbsoluteReturnsToActorRelative(r.acting_player, abs_returns);

            ContinuationRecord cr;
            cr.root_id = r.root_id;
            cr.partition = r.partition;
            cr.action = a;
            cr.replicate = k;
            cr.chance_seed = c_seed;
            cr.policy_seed = p_seed;
            cr.absolute_returns = abs_returns;
            for (int s = 0; s < 4; ++s) {
              cr.actor_relative_scaled_returns[s] = rel_returns[s] / kUtilityDivisor;
            }
            batch_records.push_back(std::move(cr));
          }
        }

        // Flush incrementally
        {
          std::lock_guard<std::mutex> lock(write_mutex);
          for (const auto& cr : batch_records) {
            json::Object obj;
            obj["root_id"] = cr.root_id;
            obj["partition"] = PartitionToString(cr.partition);
            obj["action"] = static_cast<int64_t>(cr.action);
            obj["replicate"] = static_cast<int64_t>(cr.replicate);
            obj["chance_seed"] = static_cast<int64_t>(cr.chance_seed);
            obj["policy_seed"] = static_cast<int64_t>(cr.policy_seed);

            json::Array abs_ret;
            for (double v : cr.absolute_returns) abs_ret.push_back(v);
            obj["absolute_returns"] = abs_ret;

            json::Array rel_ret;
            for (double v : cr.actor_relative_scaled_returns) rel_ret.push_back(v);
            obj["actor_relative_scaled_returns"] = rel_ret;

            out_f << json::ToString(obj) << "\n";
          }
          out_f.flush();
        }

        int done = completed.fetch_add(batch_records.size()) + batch_records.size();
        if (done % 5000 < static_cast<int>(batch_records.size()) || done == total_targets) {
          auto now = std::chrono::steady_clock::now();
          double elapsed = std::chrono::duration<double>(now - start_time).count();
          double rate = done / elapsed;
          std::cout << absl::StrFormat("  Completed %d / %d continuations (%.1f%%, %.1f conts/sec)\n",
                                       done, total_targets, 100.0 * done / total_targets, rate);
        }
      }
    });
  }

  for (auto& w : workers) w.join();
  out_f.close();
  std::cout << "TRAIN & DEV ROLLOUT COMPLETE.\n";
}

// Load continuations into map: (root_id, action) -> vector of ContinuationRecord
std::map<std::pair<std::string, Action>, std::vector<ContinuationRecord>> LoadContinuations(
    const std::filesystem::path& path) {
  std::ifstream f(path);
  if (!f.is_open()) SpielFatalError("Cannot open continuations file: " + path.string());
  std::string line;
  std::map<std::pair<std::string, Action>, std::vector<ContinuationRecord>> result;

  while (std::getline(f, line)) {
    if (line.empty()) continue;
    auto parsed = json::FromString(line);
    SPIEL_CHECK_TRUE(parsed.has_value());
    const auto& obj = parsed.value().GetObject();

    ContinuationRecord cr;
    cr.root_id = obj.at("root_id").GetString();
    cr.partition = StringToPartition(obj.at("partition").GetString());
    cr.action = static_cast<Action>(obj.at("action").GetInt());
    cr.replicate = static_cast<int>(obj.at("replicate").GetInt());
    cr.chance_seed = static_cast<uint64_t>(obj.at("chance_seed").GetInt());
    cr.policy_seed = static_cast<uint64_t>(obj.at("policy_seed").GetInt());

    int i = 0;
    for (const auto& v : obj.at("absolute_returns").GetArray()) {
      cr.absolute_returns[i++] = v.GetDouble();
    }
    i = 0;
    for (const auto& v : obj.at("actor_relative_scaled_returns").GetArray()) {
      cr.actor_relative_scaled_returns[i++] = v.GetDouble();
    }
    result[{cr.root_id, cr.action}].push_back(std::move(cr));
  }
  return result;
}

// Phase: train_critics
void RunTrainCritics(const std::filesystem::path& out_dir, const torch::Device& device) {
  std::cout << "\n=======================================================\n";
  std::cout << "PHASE: Train Matched Critics (S vs M)\n";
  std::cout << "=======================================================\n";

  auto roots = LoadCorpusRoots(out_dir / "corpus_roots.jsonl");
  auto continuations = LoadContinuations(out_dir / "train_dev_continuations.jsonl");

  std::vector<RootRecord> train_roots;
  std::vector<RootRecord> dev_roots;
  for (auto& r : roots) {
    if (r.partition == Partition::kTrain) train_roots.push_back(std::move(r));
    else if (r.partition == Partition::kDev) dev_roots.push_back(std::move(r));
  }
  SPIEL_CHECK_EQ(train_roots.size(), kRootsTrain);
  SPIEL_CHECK_EQ(dev_roots.size(), kRootsDev);

  // Compute immutable targets for each root & action:
  // For Train roots:
  //   S: replicate 0 actor-relative scaled return vector
  //   M: mean over all 16 continuations
  struct ActionTarget {
    Action action;
    std::array<double, 4> s_target; // replicate 0
    std::array<double, 4> m_target; // 16-rep mean
  };

  struct RootTargets {
    std::string root_id;
    Player acting_player;
    std::vector<float> critic_input;
    std::vector<ActionTarget> actions;
    // Dev 32-continuation mean targets for development scoring
    std::vector<std::array<double, 4>> dev_32_mean_targets;
  };

  std::vector<RootTargets> train_data;
  train_data.reserve(train_roots.size());

  for (const auto& r : train_roots) {
    RootTargets rt;
    rt.root_id = r.root_id;
    rt.acting_player = r.acting_player;
    rt.critic_input = r.critic_input_9647;

    for (Action a : r.candidate_actions) {
      const auto& reps = continuations.at({r.root_id, a});
      SPIEL_CHECK_EQ(reps.size(), kContinuationsTrain);

      ActionTarget at;
      at.action = a;
      at.s_target = reps[0].actor_relative_scaled_returns;

      std::array<double, 4> sum{};
      for (const auto& cr : reps) {
        for (int s = 0; s < 4; ++s) sum[s] += cr.actor_relative_scaled_returns[s];
      }
      for (int s = 0; s < 4; ++s) at.m_target[s] = sum[s] / reps.size();
      rt.actions.push_back(std::move(at));
    }
    train_data.push_back(std::move(rt));
  }

  std::vector<RootTargets> dev_data;
  dev_data.reserve(dev_roots.size());

  for (const auto& r : dev_roots) {
    RootTargets rt;
    rt.root_id = r.root_id;
    rt.acting_player = r.acting_player;
    rt.critic_input = r.critic_input_9647;

    for (Action a : r.candidate_actions) {
      const auto& reps = continuations.at({r.root_id, a});
      SPIEL_CHECK_EQ(reps.size(), kContinuationsDev);

      ActionTarget at;
      at.action = a;
      std::array<double, 4> sum{};
      for (const auto& cr : reps) {
        for (int s = 0; s < 4; ++s) sum[s] += cr.actor_relative_scaled_returns[s];
      }
      for (int s = 0; s < 4; ++s) sum[s] /= reps.size();
      at.m_target = sum;
      rt.actions.push_back(std::move(at));
      rt.dev_32_mean_targets.push_back(sum);
    }
    dev_data.push_back(std::move(rt));
  }

  // Pure FP32, TF32 disabled
  at::globalContext().setAllowTF32CuBLAS(false);
  at::globalContext().setAllowTF32CuDNN(false);

  // Initialize both critics from one saved, identical random parameter state with seed 19
  auto critic_s = std::make_shared<DuneVrpoQNetImpl>(kRegisteredCriticInitSeed, kCriticInputDim);
  auto critic_m = std::make_shared<DuneVrpoQNetImpl>(kRegisteredCriticInitSeed, kCriticInputDim);
  critic_s->to(device);
  critic_m->to(device);

  // Verify identical initial weights
  for (const auto& p1 : critic_s->named_parameters()) {
    const auto& p2 = critic_m->named_parameters()[p1.key()];
    SPIEL_CHECK_TRUE(torch::equal(p1.value(), p2));
  }

  torch::optim::AdamW opt_s(
      critic_s->parameters(),
      torch::optim::AdamWOptions(kCriticLearningRate)
          .eps(kCriticAdamWEpsilon)
          .weight_decay(kCriticWeightDecay));

  torch::optim::AdamW opt_m(
      critic_m->parameters(),
      torch::optim::AdamWOptions(kCriticLearningRate)
          .eps(kCriticAdamWEpsilon)
          .weight_decay(kCriticWeightDecay));

  // Development scoring evaluator:
  // Measures acting-player action-difference MSE:
  // For each dev root i:
  //   Candidate 0 is reference action a_0
  //   For alternative j in 1..K-1:
  //     pred_diff = pred(a_j)[0] - pred(a_0)[0]
  //     true_diff = target(a_j)[0] - target(a_0)[0]
  //     sq_err = (pred_diff - true_diff)^2
  //   Average sq_err across alternatives in root i, then average across roots
  auto evaluate_dev = [&](std::shared_ptr<DuneVrpoQNetImpl>& critic) -> double {
    critic->eval();
    torch::NoGradGuard no_grad;

    double total_root_diff_mse = 0.0;
    for (const auto& dr : dev_data) {
      torch::Tensor input = torch::from_blob(
          const_cast<float*>(dr.critic_input.data()),
          {1, kCriticInputDim}, torch::kFloat32).to(device);
      torch::Tensor q_out;
      std::string err;
      SPIEL_CHECK_TRUE(critic->ForwardChecked(input, &q_out, &err));
      // q_out shape: [1, 2391, 4]
      torch::Tensor q_cpu = q_out.squeeze(0).to(torch::kCPU);

      Action ref_action = dr.actions[0].action;
      double ref_pred = q_cpu[ref_action][0].item<double>();
      double ref_true = dr.dev_32_mean_targets[0][0];

      double sum_alt_err = 0.0;
      int num_alts = dr.actions.size() - 1;
      for (size_t j = 1; j < dr.actions.size(); ++j) {
        Action alt_action = dr.actions[j].action;
        double alt_pred = q_cpu[alt_action][0].item<double>();
        double alt_true = dr.dev_32_mean_targets[j][0];

        double pred_diff = alt_pred - ref_pred;
        double true_diff = alt_true - ref_true;
        double err_diff = pred_diff - true_diff;
        sum_alt_err += err_diff * err_diff;
      }
      total_root_diff_mse += sum_alt_err / num_alts;
    }
    return total_root_diff_mse / dev_data.size();
  };

  // Directory for checkpoints
  std::filesystem::path ckpt_dir = out_dir / "critic_checkpoints";
  std::filesystem::create_directories(ckpt_dir);

  struct CheckpointMeta {
    int epoch = 0;
    double dev_action_diff_mse = 0.0;
    std::string path;
    std::string sha256;
  };

  std::vector<CheckpointMeta> s_checkpoints;
  std::vector<CheckpointMeta> m_checkpoints;

  auto save_and_eval = [&](int epoch) {
    double dev_mse_s = evaluate_dev(critic_s);
    double dev_mse_m = evaluate_dev(critic_m);

    std::string path_s = (ckpt_dir / absl::StrFormat("critic_s_epoch_%03d.pt", epoch)).string();
    std::string path_m = (ckpt_dir / absl::StrFormat("critic_m_epoch_%03d.pt", epoch)).string();

    torch::serialize::OutputArchive out_s;
    critic_s->save(out_s);
    out_s.save_to(path_s);

    torch::serialize::OutputArchive out_m;
    critic_m->save(out_m);
    out_m.save_to(path_m);

    CheckpointMeta meta_s{epoch, dev_mse_s, path_s, ComputeFileSHA256(path_s)};
    CheckpointMeta meta_m{epoch, dev_mse_m, path_m, ComputeFileSHA256(path_m)};

    s_checkpoints.push_back(meta_s);
    m_checkpoints.push_back(meta_m);

    std::cout << absl::StrFormat("  Epoch %3d / 100 | Dev Action-Diff MSE: S = %.6f, M = %.6f\n",
                                 epoch, dev_mse_s, dev_mse_m);
  };

  // Record at initialization (epoch 0)
  std::cout << "Evaluating initialization (epoch 0)...\n";
  save_and_eval(0);

  // Training loop: exactly 100 epochs, matched root minibatch order (128 roots per minibatch)
  const int batch_roots = kCriticMinibatchSizeRoots; // 128
  const int num_batches = (train_data.size() + batch_roots - 1) / batch_roots; // 16 batches

  for (int epoch = 1; epoch <= kCriticTotalEpochs; ++epoch) {
    critic_s->train();
    critic_m->train();

    // Minibatch order is identical for both critics
    for (int b = 0; b < num_batches; ++b) {
      int start_idx = b * batch_roots;
      int end_idx = std::min(start_idx + batch_roots, static_cast<int>(train_data.size()));
      int cur_batch_size = end_idx - start_idx;

      // Pack inputs
      torch::Tensor batch_inputs = torch::empty({cur_batch_size, kCriticInputDim}, torch::kFloat32);
      for (int i = 0; i < cur_batch_size; ++i) {
        std::memcpy(batch_inputs[i].data_ptr<float>(),
                    train_data[start_idx + i].critic_input.data(),
                    kCriticInputDim * sizeof(float));
      }
      batch_inputs = batch_inputs.to(device);

      // --- CRITIC S STEP ---
      {
        torch::Tensor q_out_s;
        std::string err;
        SPIEL_CHECK_TRUE(critic_s->ForwardChecked(batch_inputs, &q_out_s, &err));

        torch::Tensor loss_s = torch::zeros({}, torch::kFloat32).to(device);
        for (int i = 0; i < cur_batch_size; ++i) {
          const auto& rt = train_data[start_idx + i];
          torch::Tensor root_err = torch::zeros({}, torch::kFloat32).to(device);
          for (const auto& at : rt.actions) {
            torch::Tensor target = torch::tensor(
                {static_cast<float>(at.s_target[0]), static_cast<float>(at.s_target[1]),
                 static_cast<float>(at.s_target[2]), static_cast<float>(at.s_target[3])},
                torch::kFloat32).to(device);
            torch::Tensor pred = q_out_s[i][at.action];
            root_err = root_err + torch::mse_loss(pred, target);
          }
          loss_s = loss_s + (root_err / static_cast<float>(rt.actions.size()));
        }
        loss_s = loss_s / static_cast<float>(cur_batch_size);

        opt_s.zero_grad();
        loss_s.backward();
        torch::nn::utils::clip_grad_norm_(critic_s->parameters(), kCriticGradClipNorm);
        opt_s.step();
      }

      // --- CRITIC M STEP ---
      {
        torch::Tensor q_out_m;
        std::string err;
        SPIEL_CHECK_TRUE(critic_m->ForwardChecked(batch_inputs, &q_out_m, &err));

        torch::Tensor loss_m = torch::zeros({}, torch::kFloat32).to(device);
        for (int i = 0; i < cur_batch_size; ++i) {
          const auto& rt = train_data[start_idx + i];
          torch::Tensor root_err = torch::zeros({}, torch::kFloat32).to(device);
          for (const auto& at : rt.actions) {
            torch::Tensor target = torch::tensor(
                {static_cast<float>(at.m_target[0]), static_cast<float>(at.m_target[1]),
                 static_cast<float>(at.m_target[2]), static_cast<float>(at.m_target[3])},
                torch::kFloat32).to(device);
            torch::Tensor pred = q_out_m[i][at.action];
            root_err = root_err + torch::mse_loss(pred, target);
          }
          loss_m = loss_m + (root_err / static_cast<float>(rt.actions.size()));
        }
        loss_m = loss_m / static_cast<float>(cur_batch_size);

        opt_m.zero_grad();
        loss_m.backward();
        torch::nn::utils::clip_grad_norm_(critic_m->parameters(), kCriticGradClipNorm);
        opt_m.step();
      }
    }

    if (epoch % kCriticEvalIntervalEpochs == 0) {
      save_and_eval(epoch);
    }
  }

  // Select best checkpoints by lowest dev acting-player action-difference MSE, ties to earlier epoch
  auto select_best = [](const std::vector<CheckpointMeta>& list) -> CheckpointMeta {
    SPIEL_CHECK_FALSE(list.empty());
    CheckpointMeta best = list[0];
    for (size_t i = 1; i < list.size(); ++i) {
      if (list[i].dev_action_diff_mse < best.dev_action_diff_mse) {
        best = list[i];
      }
    }
    return best;
  };

  CheckpointMeta best_s = select_best(s_checkpoints);
  CheckpointMeta best_m = select_best(m_checkpoints);

  std::cout << "\n=== FREEZING SELECTED CRITIC CHECKPOINTS ===\n";
  std::cout << absl::StrFormat("Selected Critic S: Epoch %d, Dev Action-Diff MSE = %.6f, SHA256 = %s\n",
                               best_s.epoch, best_s.dev_action_diff_mse, best_s.sha256);
  std::cout << absl::StrFormat("Selected Critic M: Epoch %d, Dev Action-Diff MSE = %.6f, SHA256 = %s\n",
                               best_m.epoch, best_m.dev_action_diff_mse, best_m.sha256);

  json::Object selected_obj;
  json::Object obj_s;
  obj_s["epoch"] = static_cast<int64_t>(best_s.epoch);
  obj_s["dev_action_diff_mse"] = best_s.dev_action_diff_mse;
  obj_s["path"] = best_s.path;
  obj_s["sha256"] = best_s.sha256;
  selected_obj["critic_s"] = obj_s;

  json::Object obj_m;
  obj_m["epoch"] = static_cast<int64_t>(best_m.epoch);
  obj_m["dev_action_diff_mse"] = best_m.dev_action_diff_mse;
  obj_m["path"] = best_m.path;
  obj_m["sha256"] = best_m.sha256;
  selected_obj["critic_m"] = obj_m;

  std::ofstream sel_f(out_dir / "selected_checkpoints.json");
  sel_f << json::ToString(selected_obj) << "\n";
  sel_f.close();
  std::cout << "CHECKPOINT SELECTION FROZEN SUCCESSFULLY.\n";
}

// Phase: rollout_test
void RunRolloutTest(const std::shared_ptr<const Game>& game, const std::filesystem::path& out_dir, int num_threads) {
  std::cout << "\n=======================================================\n";
  std::cout << "PHASE: Rollout Final Test Partition (64 Continuations per Action)\n";
  std::cout << "=======================================================\n";

  // Check that checkpoint selection is frozen
  std::filesystem::path sel_path = out_dir / "selected_checkpoints.json";
  if (!std::filesystem::exists(sel_path)) {
    SpielFatalError("Cannot run test rollouts before selected_checkpoints.json is frozen!");
  }

  auto roots = LoadCorpusRoots(out_dir / "corpus_roots.jsonl");
  std::filesystem::path conts_path = out_dir / "test_continuations.jsonl";
  std::ofstream out_f(conts_path, std::ios::app);

  std::atomic<int> completed{0};
  int total_test_conts = 0;
  for (const auto& r : roots) {
    if (r.partition == Partition::kTest) {
      total_test_conts += r.candidate_actions.size() * kContinuationsTest;
    }
  }
  std::cout << "Total test continuations to run: " << total_test_conts << "\n";

  auto start_time = std::chrono::steady_clock::now();
  std::mutex write_mutex;

  std::atomic<size_t> next_root_idx{0};
  std::vector<std::thread> workers;

  for (int tid = 0; tid < num_threads; ++tid) {
    workers.emplace_back([&]() {
      while (true) {
        size_t idx = next_root_idx.fetch_add(1);
        if (idx >= roots.size()) break;
        const auto& r = roots[idx];
        if (r.partition != Partition::kTest) continue;

        std::vector<ContinuationRecord> batch_records;
        for (Action a : r.candidate_actions) {
          for (int k = 0; k < kContinuationsTest; ++k) {
            uint64_t c_seed = DeriveContinuationChanceSeed(Partition::kTest, r.root_id, k);
            uint64_t p_seed = DeriveContinuationPolicySeed(Partition::kTest, r.root_id, k);

            auto state = ReconstructState(game, r.history);
            state->ApplyAction(a);
            auto abs_returns = RunRollout(game, std::move(state), c_seed, p_seed);
            auto rel_returns = ConvertAbsoluteReturnsToActorRelative(r.acting_player, abs_returns);

            ContinuationRecord cr;
            cr.root_id = r.root_id;
            cr.partition = Partition::kTest;
            cr.action = a;
            cr.replicate = k;
            cr.chance_seed = c_seed;
            cr.policy_seed = p_seed;
            cr.absolute_returns = abs_returns;
            for (int s = 0; s < 4; ++s) {
              cr.actor_relative_scaled_returns[s] = rel_returns[s] / kUtilityDivisor;
            }
            batch_records.push_back(std::move(cr));
          }
        }

        {
          std::lock_guard<std::mutex> lock(write_mutex);
          for (const auto& cr : batch_records) {
            json::Object obj;
            obj["root_id"] = cr.root_id;
            obj["partition"] = PartitionToString(cr.partition);
            obj["action"] = static_cast<int64_t>(cr.action);
            obj["replicate"] = static_cast<int64_t>(cr.replicate);
            obj["chance_seed"] = static_cast<int64_t>(cr.chance_seed);
            obj["policy_seed"] = static_cast<int64_t>(cr.policy_seed);

            json::Array abs_ret;
            for (double v : cr.absolute_returns) abs_ret.push_back(v);
            obj["absolute_returns"] = abs_ret;

            json::Array rel_ret;
            for (double v : cr.actor_relative_scaled_returns) rel_ret.push_back(v);
            obj["actor_relative_scaled_returns"] = rel_ret;

            out_f << json::ToString(obj) << "\n";
          }
          out_f.flush();
        }

        int done = completed.fetch_add(batch_records.size()) + batch_records.size();
        if (done % 5000 < static_cast<int>(batch_records.size()) || done == total_test_conts) {
          auto now = std::chrono::steady_clock::now();
          double elapsed = std::chrono::duration<double>(now - start_time).count();
          double rate = done / elapsed;
          std::cout << absl::StrFormat("  Completed %d / %d test continuations (%.1f%%, %.1f conts/sec)\n",
                                       done, total_test_conts, 100.0 * done / total_test_conts, rate);
        }
      }
    });
  }

  for (auto& w : workers) w.join();
  out_f.close();
  std::cout << "TEST ROLLOUT COMPLETE.\n";
}

// Phase: evaluate
void RunEvaluate(const std::filesystem::path& out_dir, const torch::Device& device) {
  std::cout << "\n=======================================================\n";
  std::cout << "PHASE: Held-Out Test Evaluation & Fixed Verdict\n";
  std::cout << "=======================================================\n";

  auto roots = LoadCorpusRoots(out_dir / "corpus_roots.jsonl");
  auto continuations = LoadContinuations(out_dir / "test_continuations.jsonl");

  std::vector<RootRecord> test_roots;
  for (auto& r : roots) {
    if (r.partition == Partition::kTest) test_roots.push_back(std::move(r));
  }
  SPIEL_CHECK_EQ(test_roots.size(), kRootsTest);

  // Load frozen selected checkpoints
  std::ifstream sel_f(out_dir / "selected_checkpoints.json");
  if (!sel_f.is_open()) SpielFatalError("Cannot open selected_checkpoints.json");
  std::string sel_text((std::istreambuf_iterator<char>(sel_f)), std::istreambuf_iterator<char>());
  auto sel_json = json::FromString(sel_text).value().GetObject();

  std::string path_s = sel_json.at("critic_s").GetObject().at("path").GetString();
  std::string path_m = sel_json.at("critic_m").GetObject().at("path").GetString();

  auto critic_s = std::make_shared<DuneVrpoQNetImpl>(0, kCriticInputDim);
  torch::serialize::InputArchive in_s;
  in_s.load_from(path_s);
  critic_s->load(in_s);
  critic_s->to(device);
  critic_s->eval();
  SPIEL_CHECK_TRUE(critic_s->CheckSchema());

  auto critic_m = std::make_shared<DuneVrpoQNetImpl>(0, kCriticInputDim);
  torch::serialize::InputArchive in_m;
  in_m.load_from(path_m);
  critic_m->load(in_m);
  critic_m->to(device);
  critic_m->eval();
  SPIEL_CHECK_TRUE(critic_m->CheckSchema());

  at::globalContext().setAllowTF32CuBLAS(false);
  at::globalContext().setAllowTF32CuDNN(false);
  torch::NoGradGuard no_grad;

  // 1. Action Difference Evaluation
  // Per-root errors for M, S, and Zero baseline
  std::vector<double> root_err_m(test_roots.size(), 0.0);
  std::vector<double> root_err_s(test_roots.size(), 0.0);
  std::vector<double> root_err_zero(test_roots.size(), 0.0);
  std::vector<double> paired_err_diff_m_minus_zero(test_roots.size(), 0.0);
  std::vector<double> paired_err_diff_m_minus_s(test_roots.size(), 0.0);

  // Local usefulness tracking
  std::vector<double> per_root_utility_gain(test_roots.size(), 0.0);
  int changed_count = 0;
  std::map<int, int> changed_by_round;
  std::map<int, int> changed_by_seat;

  // Target reliability tracking (first 32 vs second 32 continuations)
  std::vector<double> half1_action_diffs;
  std::vector<double> half2_action_diffs;

  for (size_t i = 0; i < test_roots.size(); ++i) {
    const auto& r = test_roots[i];
    torch::Tensor input = torch::from_blob(
        const_cast<float*>(r.critic_input_9647.data()),
        {1, kCriticInputDim}, torch::kFloat32).to(device);

    torch::Tensor q_out_m, q_out_s;
    std::string err;
    SPIEL_CHECK_TRUE(critic_m->ForwardChecked(input, &q_out_m, &err));
    SPIEL_CHECK_TRUE(critic_s->ForwardChecked(input, &q_out_s, &err));

    torch::Tensor qm_cpu = q_out_m.squeeze(0).to(torch::kCPU);
    torch::Tensor qs_cpu = q_out_s.squeeze(0).to(torch::kCPU);

    Action ref_action = r.candidate_actions[0];
    const auto& ref_conts = continuations.at({r.root_id, ref_action});
    SPIEL_CHECK_EQ(ref_conts.size(), kContinuationsTest);

    // Compute reference true mean return across all 64 continuations (acting player)
    double sum_ref = 0.0;
    double sum_ref_h1 = 0.0;
    double sum_ref_h2 = 0.0;
    for (int k = 0; k < kContinuationsTest; ++k) {
      double ret_val = ref_conts[k].actor_relative_scaled_returns[0] * kUtilityDivisor;
      sum_ref += ret_val;
      if (k < 32) sum_ref_h1 += ret_val;
      else sum_ref_h2 += ret_val;
    }
    double true_ref_mean = sum_ref / kContinuationsTest;
    double true_ref_h1 = sum_ref_h1 / 32.0;
    double true_ref_h2 = sum_ref_h2 / 32.0;

    double pred_ref_m = qm_cpu[ref_action][0].item<double>() * kUtilityDivisor;
    double pred_ref_s = qs_cpu[ref_action][0].item<double>() * kUtilityDivisor;

    int num_alts = r.candidate_actions.size() - 1;
    double root_sum_err_m = 0.0;
    double root_sum_err_s = 0.0;
    double root_sum_err_zero = 0.0;

    // Track best action according to Critic M
    Action best_action_m = ref_action;
    double best_q_m = pred_ref_m;

    for (size_t j = 1; j < r.candidate_actions.size(); ++j) {
      Action alt_action = r.candidate_actions[j];
      const auto& alt_conts = continuations.at({r.root_id, alt_action});
      SPIEL_CHECK_EQ(alt_conts.size(), kContinuationsTest);

      double sum_alt = 0.0;
      double sum_alt_h1 = 0.0;
      double sum_alt_h2 = 0.0;
      for (int k = 0; k < kContinuationsTest; ++k) {
        double ret_val = alt_conts[k].actor_relative_scaled_returns[0] * kUtilityDivisor;
        sum_alt += ret_val;
        if (k < 32) sum_alt_h1 += ret_val;
        else sum_alt_h2 += ret_val;
      }
      double true_alt_mean = sum_alt / kContinuationsTest;
      double true_alt_h1 = sum_alt_h1 / 32.0;
      double true_alt_h2 = sum_alt_h2 / 32.0;

      double pred_alt_m = qm_cpu[alt_action][0].item<double>() * kUtilityDivisor;
      double pred_alt_s = qs_cpu[alt_action][0].item<double>() * kUtilityDivisor;

      // Predicted differences vs true differences
      double true_diff = true_alt_mean - true_ref_mean;
      double pred_diff_m = pred_alt_m - pred_ref_m;
      double pred_diff_s = pred_alt_s - pred_ref_s;
      double pred_diff_zero = 0.0;

      double diff_err_m = pred_diff_m - true_diff;
      double diff_err_s = pred_diff_s - true_diff;
      double diff_err_zero = pred_diff_zero - true_diff;

      root_sum_err_m += diff_err_m * diff_err_m;
      root_sum_err_s += diff_err_s * diff_err_s;
      root_sum_err_zero += diff_err_zero * diff_err_zero;

      // Check Critic M selection (ties favouring reference action)
      if (pred_alt_m > best_q_m) {
        best_q_m = pred_alt_m;
        best_action_m = alt_action;
      }

      // Target reliability pairs
      half1_action_diffs.push_back(true_alt_h1 - true_ref_h1);
      half2_action_diffs.push_back(true_alt_h2 - true_ref_h2);
    }

    root_err_m[i] = root_sum_err_m / num_alts;
    root_err_s[i] = root_sum_err_s / num_alts;
    root_err_zero[i] = root_sum_err_zero / num_alts;
    paired_err_diff_m_minus_zero[i] = root_err_m[i] - root_err_zero[i];
    paired_err_diff_m_minus_s[i] = root_err_m[i] - root_err_s[i];

    // Local usefulness for root i:
    // Score chosen action vs raw-reference action on independent continuations
    if (best_action_m != ref_action) {
      changed_count++;
      changed_by_round[r.round]++;
      changed_by_seat[r.acting_player]++;
      const auto& chosen_conts = continuations.at({r.root_id, best_action_m});
      double sum_chosen = 0.0;
      for (int k = 0; k < kContinuationsTest; ++k) {
        sum_chosen += chosen_conts[k].actor_relative_scaled_returns[0] * kUtilityDivisor;
      }
      double true_chosen_mean = sum_chosen / kContinuationsTest;
      per_root_utility_gain[i] = true_chosen_mean - true_ref_mean;
    } else {
      per_root_utility_gain[i] = 0.0;
    }
  }

  // Aggregate Metrics across all test roots
  double mean_mse_m = std::accumulate(root_err_m.begin(), root_err_m.end(), 0.0) / test_roots.size();
  double mean_mse_s = std::accumulate(root_err_s.begin(), root_err_s.end(), 0.0) / test_roots.size();
  double mean_mse_zero = std::accumulate(root_err_zero.begin(), root_err_zero.end(), 0.0) / test_roots.size();

  double red_m_vs_s = (mean_mse_s > 0.0) ? (mean_mse_s - mean_mse_m) / mean_mse_s : 0.0;
  double red_m_vs_zero = (mean_mse_zero > 0.0) ? (mean_mse_zero - mean_mse_m) / mean_mse_zero : 0.0;
  double red_s_vs_zero = (mean_mse_zero > 0.0) ? (mean_mse_zero - mean_mse_s) / mean_mse_zero : 0.0;

  auto ci_m_minus_zero = Bootstrap95CI(paired_err_diff_m_minus_zero, kBootstrapResamples, kBootstrapSeed);
  auto ci_m_minus_s = Bootstrap95CI(paired_err_diff_m_minus_s, kBootstrapResamples, kBootstrapSeed + 1);

  double mean_util_gain = std::accumulate(per_root_utility_gain.begin(), per_root_utility_gain.end(), 0.0) / test_roots.size();
  auto ci_util_gain = Bootstrap95CI(per_root_utility_gain, kBootstrapResamples, kBootstrapSeed + 2);
  double changed_frac = static_cast<double>(changed_count) / test_roots.size();

  // Target reliability correlation
  double mean_h1 = 0.0, mean_h2 = 0.0;
  for (size_t i = 0; i < half1_action_diffs.size(); ++i) {
    mean_h1 += half1_action_diffs[i];
    mean_h2 += half2_action_diffs[i];
  }
  mean_h1 /= half1_action_diffs.size();
  mean_h2 /= half2_action_diffs.size();

  double cov = 0.0, var1 = 0.0, var2 = 0.0;
  double sum_abs_diff = 0.0, sum_sq_diff = 0.0;
  for (size_t i = 0; i < half1_action_diffs.size(); ++i) {
    double d1 = half1_action_diffs[i] - mean_h1;
    double d2 = half2_action_diffs[i] - mean_h2;
    cov += d1 * d2;
    var1 += d1 * d1;
    var2 += d2 * d2;
    double diff = half1_action_diffs[i] - half2_action_diffs[i];
    sum_abs_diff += std::abs(diff);
    sum_sq_diff += diff * diff;
  }
  double corr = (var1 > 0.0 && var2 > 0.0) ? (cov / std::sqrt(var1 * var2)) : 0.0;
  double mean_abs_diff = sum_abs_diff / half1_action_diffs.size();
  double agreement_mse = sum_sq_diff / half1_action_diffs.size();

  // Advancement Criteria Evaluation
  bool pass_criterion_1 = (red_m_vs_s >= 0.10) && (red_m_vs_zero >= 0.10) &&
                          (ci_m_minus_zero.second < 0.0) && (ci_m_minus_s.second < 0.0);
  bool pass_criterion_2 = (mean_util_gain >= 0.05) && (ci_util_gain.first > 0.0);

  StudyVerdict verdict;
  if (pass_criterion_1 && pass_criterion_2) {
    verdict = StudyVerdict::kPassQLearningScreen;
  } else {
    verdict = StudyVerdict::kNoDemonstratedUsefulQGain;
  }

  // Print Formatted Table (Strictly aligned, no LaTeX, <100 width)
  std::cout << "\n=========================================================================\n";
  std::cout << "                 ACTION-VALUE LEARNABILITY STUDY REPORT                  \n";
  std::cout << "=========================================================================\n";
  std::cout << absl::StrFormat("Verdict: %s\n\n", VerdictToString(verdict));

  std::cout << "Action-Difference Error (MSE on unseen test roots):\n";
  std::cout << "+-----------------------+----------+--------------------+----------------------+\n";
  std::cout << "| Predictor             | MSE      | Relative Reduction | Paired 95% CI vs M   |\n";
  std::cout << "+-----------------------+----------+--------------------+----------------------+\n";
  std::cout << absl::StrFormat("| M (16-mean target)    | %8.4f |      reference     |      reference       |\n", mean_mse_m);
  std::cout << absl::StrFormat("| S (single target)     | %8.4f | %6.2f%% vs S       | [%7.4f, %7.4f] |\n",
                               mean_mse_s, red_m_vs_s * 100.0, ci_m_minus_s.first, ci_m_minus_s.second);
  std::cout << absl::StrFormat("| Zero-diff baseline    | %8.4f | %6.2f%% vs Zero    | [%7.4f, %7.4f] |\n",
                               mean_mse_zero, red_m_vs_zero * 100.0, ci_m_minus_zero.first, ci_m_minus_zero.second);
  std::cout << "+-----------------------+----------+--------------------+----------------------+\n\n";

  std::cout << "Local Decision Usefulness (Critic M vs Frozen Actor Reference):\n";
  std::cout << absl::StrFormat("  Mean Utility Gain     : %+7.4f (unscaled utility)\n", mean_util_gain);
  std::cout << absl::StrFormat("  95%% Confidence Bound  : [%+7.4f, %+7.4f]\n", ci_util_gain.first, ci_util_gain.second);
  std::cout << absl::StrFormat("  Action Changed Rate   : %5.2f%% (%d / %d roots)\n\n",
                               changed_frac * 100.0, changed_count, static_cast<int>(test_roots.size()));

  std::cout << "Target Reliability (First 32 vs Second 32 Continuations):\n";
  std::cout << absl::StrFormat("  Pearson Correlation r : %7.4f\n", corr);
  std::cout << absl::StrFormat("  Mean Absolute Diff    : %7.4f\n", mean_abs_diff);
  std::cout << absl::StrFormat("  Agreement MSE         : %7.4f\n\n", agreement_mse);

  std::cout << "Advancement Criteria Check:\n";
  std::cout << absl::StrFormat("  Criterion 1 (MSE >= 10%% lower than S & Zero, CI < 0): %s\n",
                               pass_criterion_1 ? "PASS" : "FAIL");
  std::cout << absl::StrFormat("  Criterion 2 (Gain >= +0.05, 95%% lower CI > 0)       : %s\n",
                               pass_criterion_2 ? "PASS" : "FAIL");
  std::cout << "=========================================================================\n";

  // Write JSON report
  json::Object final_rep;
  final_rep["verdict"] = VerdictToString(verdict);
  final_rep["criterion_1_passed"] = pass_criterion_1;
  final_rep["criterion_2_passed"] = pass_criterion_2;

  json::Object mse_obj;
  mse_obj["mse_m"] = mean_mse_m;
  mse_obj["mse_s"] = mean_mse_s;
  mse_obj["mse_zero"] = mean_mse_zero;
  mse_obj["reduction_m_vs_s"] = red_m_vs_s;
  mse_obj["reduction_m_vs_zero"] = red_m_vs_zero;
  mse_obj["reduction_s_vs_zero"] = red_s_vs_zero;
  mse_obj["ci95_m_minus_zero_low"] = ci_m_minus_zero.first;
  mse_obj["ci95_m_minus_zero_high"] = ci_m_minus_zero.second;
  mse_obj["ci95_m_minus_s_low"] = ci_m_minus_s.first;
  mse_obj["ci95_m_minus_s_high"] = ci_m_minus_s.second;
  final_rep["action_diff_mse"] = mse_obj;

  json::Object util_obj;
  util_obj["mean_utility_gain"] = mean_util_gain;
  util_obj["ci95_gain_low"] = ci_util_gain.first;
  util_obj["ci95_gain_high"] = ci_util_gain.second;
  util_obj["changed_choice_fraction"] = changed_frac;
  util_obj["changed_count"] = static_cast<int64_t>(changed_count);
  util_obj["total_test_roots"] = static_cast<int64_t>(test_roots.size());

  json::Object r_breakdown;
  for (const auto& kv : changed_by_round) {
    r_breakdown[std::to_string(kv.first)] = static_cast<int64_t>(kv.second);
  }
  util_obj["changed_by_round"] = r_breakdown;

  json::Object s_breakdown;
  for (const auto& kv : changed_by_seat) {
    s_breakdown[std::to_string(kv.first)] = static_cast<int64_t>(kv.second);
  }
  util_obj["changed_by_seat"] = s_breakdown;
  final_rep["local_usefulness"] = util_obj;

  json::Object rel_obj;
  rel_obj["pearson_correlation"] = corr;
  rel_obj["mean_abs_diff"] = mean_abs_diff;
  rel_obj["agreement_mse"] = agreement_mse;
  final_rep["target_reliability"] = rel_obj;

  std::ofstream rep_f(out_dir / "final_report.json");
  rep_f << json::ToString(final_rep) << "\n";
  rep_f.close();
}

}  // namespace action_value_study
}  // namespace open_spiel

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  using namespace open_spiel;
  using namespace open_spiel::action_value_study;

  std::string phase = absl::GetFlag(FLAGS_phase);
  std::string out_dir_str = absl::GetFlag(FLAGS_output_dir);
  if (phase != "verify" && out_dir_str.empty()) {
    SpielFatalError("--output_dir is required for phase: " + phase);
  }

  std::filesystem::path out_dir(out_dir_str);
  if (!out_dir.empty()) {
    std::filesystem::create_directories(out_dir);
  }

  auto game = LoadGame("dune_imperium");
  torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
  std::cout << "Active compute device: " << (device.is_cuda() ? "CUDA (RTX 4080)" : "CPU") << std::endl;

  int threads = absl::GetFlag(FLAGS_threads);
  int batch_size = absl::GetFlag(FLAGS_eval_batch_size);
  int timeout_ms = absl::GetFlag(FLAGS_eval_timeout_ms);

  if (phase == "verify") {
    RunVerify();
    return 0;
  }

  // For other phases, initialize the frozen actor
  InitializeActor(absl::GetFlag(FLAGS_model_path), device, batch_size, timeout_ms);

  if (phase == "pilot") {
    RunVerify();
    RunPilot(game, out_dir, threads);
  } else if (phase == "corpus") {
    RunGenerateCorpus(game, out_dir, threads);
  } else if (phase == "rollout_train_dev") {
    RunRolloutTrainDev(game, out_dir, threads);
  } else if (phase == "train") {
    RunTrainCritics(out_dir, device);
  } else if (phase == "rollout_test") {
    RunRolloutTest(game, out_dir, threads);
  } else if (phase == "evaluate") {
    RunEvaluate(out_dir, device);
  } else if (phase == "all") {
    RunVerify();
    RunPilot(game, out_dir, threads);
    RunGenerateCorpus(game, out_dir, threads);
    RunRolloutTrainDev(game, out_dir, threads);
    RunTrainCritics(out_dir, device);
    RunRolloutTest(game, out_dir, threads);
    RunEvaluate(out_dir, device);
  } else {
    SpielFatalError("Unknown phase: " + phase);
  }

  return 0;
}
