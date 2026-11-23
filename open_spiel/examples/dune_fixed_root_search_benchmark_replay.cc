#include "open_spiel/examples/dune_fixed_root_search_benchmark_internal.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>
#include <unordered_map>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/examples/dune_seed_utils.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel::dune_imperium::fixed_root_benchmark {

int RunHeldOutReplayEvaluation(
    const std::string& model_path,
    const std::string& trajectory_jsonl_path,
    const std::string& obs_bin_path,
    const std::string& output_json_path) {
  std::cout << "================================================================================\n";
  std::cout << "HELD-OUT TRAJECTORY REPLAY POLICY EVALUATION\n";
  std::cout << "================================================================================\n";
  std::cout << "Model checkpoint:       " << model_path << "\n";
  std::cout << "Replay trajectory meta: " << trajectory_jsonl_path << "\n";
  std::cout << "Replay observation bin: " << obs_bin_path << "\n";
  if (!output_json_path.empty()) {
    std::cout << "Output JSON receipt:    " << output_json_path << "\n";
  }

  // Load Game
  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("dune_imperium");

  // Open observation binary file
  std::ifstream obs_file(obs_bin_path, std::ios::binary);
  if (!obs_file.is_open()) {
    open_spiel::SpielFatalError("Failed to open observation binary: " + obs_bin_path);
  }
  const size_t obs_elem_count = open_spiel::dune_imperium::kFullPublicInformationStateSize;
  const size_t obs_byte_stride = obs_elem_count * sizeof(float);

  // Open trajectory JSONL
  std::ifstream meta_file(trajectory_jsonl_path);
  if (!meta_file.is_open()) {
    open_spiel::SpielFatalError("Failed to open trajectory meta jsonl: " + trajectory_jsonl_path);
  }

  // Setup Device & Model Evaluator
  EvaluatorBundle bundle = InitEvaluatorBundle(
      /*num_evals=*/1, /*target_batch_size=*/64, absl::GetFlag(FLAGS_num_blocks),
      model_path, /*rollout_amp=*/true, /*allow_tf32=*/true,
      /*disable_global_tf32=*/false, /*log_device=*/true);
  auto& coord = bundle.coords[0];
  auto& evaluator = bundle.evaluators[0];
  std::cout << "[MODEL] Successfully initialized policy evaluator with semantic scoring.\n";

  HeldOutReplayStats stats;
  std::string line;
  std::vector<float> obs_buffer(obs_elem_count);

  while (std::getline(meta_file, line)) {
    if (line.empty()) continue;
    ++stats.total_rows_read;
    auto json_opt = open_spiel::json::FromString(line);
    if (!json_opt || !json_opt->IsObject()) continue;
    const auto& obj = json_opt->GetObject();

    // Filter strictly to decision_role == "fresh_search"
    std::string decision_role = "";
    auto it_role = obj.find("decision_role");
    if (it_role != obj.end() && it_role->second.IsString()) {
      decision_role = it_role->second.GetString();
    }
    if (decision_role != "fresh_search") continue;

    int search_seat = static_cast<int>(obj.at("search_seat").GetInt());
    int round = static_cast<int>(obj.at("round").GetInt());
    int64_t obs_row = obj.at("obs_row").GetInt();

    // Parse legal_actions
    std::vector<Action> legal_actions;
    if (obj.find("legal_actions") != obj.end() && obj.at("legal_actions").IsArray()) {
      for (const auto& a_val : obj.at("legal_actions").GetArray()) {
        legal_actions.push_back(static_cast<Action>(a_val.GetInt()));
      }
    }
    if (legal_actions.empty()) continue;

    // Parse recorded mu (U28700 raw prior)
    std::vector<std::pair<Action, double>> raw_policy;
    std::unordered_map<Action, double> mu_map;
    Action argmax_mu = -1;
    double max_mu = -1.0;
    if (obj.find("mu") != obj.end() && obj.at("mu").IsObject()) {
      for (const auto& kv : obj.at("mu").GetObject()) {
        Action a = static_cast<Action>(std::stoi(kv.first));
        double prob = kv.second.GetDouble();
        raw_policy.push_back({a, prob});
        mu_map[a] = prob;
        if (prob > max_mu) {
          max_mu = prob;
          argmax_mu = a;
        }
      }
    }
    if (raw_policy.empty() || argmax_mu == -1) continue;

    // Parse candidate_utilities
    std::vector<std::pair<Action, double>> candidate_utilities;
    std::unordered_map<Action, double> cand_util_map;
    if (obj.find("candidate_utilities") != obj.end() && obj.at("candidate_utilities").IsArray()) {
      for (const auto& item : obj.at("candidate_utilities").GetArray()) {
        if (item.IsArray() && item.GetArray().size() >= 2) {
          Action a = static_cast<Action>(item.GetArray()[0].GetInt());
          double u = item.GetArray()[1].GetDouble();
          candidate_utilities.push_back({a, u});
          cand_util_map[a] = u;
        }
      }
    }
    if (candidate_utilities.empty()) continue;

    // Build target with ComputeMpoRelativeTarget(mu, legal_actions, candidate_utilities, 0.50, 0.05)
    std::vector<double> target_dist = ComputeMpoRelativeTarget(
        raw_policy, legal_actions, candidate_utilities, /*base_temperature=*/0.50, /*max_target_kl=*/0.05);
    if (target_dist.size() != legal_actions.size()) continue;

    // Recompute override states from candidate_utilities with the 0.15 rule (best candidate beats raw by >= 0.15)
    double raw_u = obj.at("raw_mean_utility").GetDouble();
    Action raw_a = static_cast<Action>(obj.at("raw_action").GetInt());
    Action best_cand_action = raw_a;
    double best_cand_utility = -1e9;
    for (const auto& cu : candidate_utilities) {
      if (cu.second > best_cand_utility) {
        best_cand_utility = cu.second;
        best_cand_action = cu.first;
      }
    }
    bool is_override = (best_cand_action != raw_a) && (best_cand_utility >= raw_u + 0.15);

    // Read observation from disk
    obs_file.seekg(obs_row * obs_byte_stride, std::ios::beg);
    obs_file.read(reinterpret_cast<char*>(obs_buffer.data()), obs_byte_stride);
    if (!obs_file) {
      open_spiel::SpielFatalError(absl::StrFormat("Failed to read obs row %d from %s", obs_row, obs_bin_path));
    }

    // Replay state from action history
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    if (obj.find("action_history") != obj.end() && obj.at("action_history").IsArray()) {
      for (const auto& a_val : obj.at("action_history").GetArray()) {
        state->ApplyAction(static_cast<Action>(a_val.GetInt()));
      }
    }
    if (state->CurrentPlayer() != search_seat) {
      open_spiel::SpielFatalError(absl::StrFormat(
          "Trajectory replay seat mismatch at row %lld: state->CurrentPlayer()=%d, search_seat=%d",
          static_cast<long long>(obs_row), state->CurrentPlayer(), search_seat));
    }

    // Verify observation consistency between disk and replayed state
    std::vector<float> recomputed_obs = evaluator->GetConsumedObservation(*state, search_seat);
    float obs_diff = 0.0f;
    for (size_t k = 0; k < obs_elem_count; ++k) {
      obs_diff = std::max(obs_diff, std::abs(recomputed_obs[k] - obs_buffer[k]));
    }
    if (obs_diff > 1e-4f) {
      open_spiel::SpielFatalError(absl::StrFormat(
          "Observation mismatch at row %lld: max diff = %f",
          static_cast<long long>(obs_row), obs_diff));
    }

    // Query policy via model and apply temperature to capped logits
    dune_semantic::CandidateActionData cand_data;
    const auto* dune = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
    if (dune != nullptr) {
      dune_semantic::ExtractCandidateDescriptors(*dune, legal_actions, &cand_data, dune_semantic::kDescriptorSchemaVersionV3);
    }
    open_spiel::EvalResult full = coord->EvaluateWithActions(recomputed_obs, &cand_data);
    std::vector<float> capped_logits = full.logits;
    open_spiel::CenterAndCapLegalLogitsWithStats(capped_logits, legal_actions, 10.0f);

    double temp = absl::GetFlag(FLAGS_temperature);
    if (temp <= 0.0) temp = 1.0;

    double max_l = -1e9;
    for (Action a : legal_actions) {
      double l = static_cast<double>(capped_logits[a]) / temp;
      if (l > max_l) max_l = l;
    }
    double sum_exp = 0.0;
    for (Action a : legal_actions) {
      sum_exp += std::exp(static_cast<double>(capped_logits[a]) / temp - max_l);
    }
    std::unordered_map<Action, double> model_probs;
    Action model_argmax = -1;
    double max_p = -1.0;
    double policy_entropy = 0.0;
    for (Action a : legal_actions) {
      double p = std::exp(static_cast<double>(capped_logits[a]) / temp - max_l) / sum_exp;
      model_probs[a] = p;
      if (p > max_p) {
        max_p = p;
        model_argmax = a;
      }
      if (p > 1e-15) {
        policy_entropy -= p * std::log(p);
      }
    }

    AccumulateHeldOutReplayDecision(
        round, obs_row, is_override, raw_a, best_cand_action, argmax_mu, model_argmax,
        policy_entropy, legal_actions, capped_logits, model_probs, mu_map,
        candidate_utilities, cand_util_map, target_dist, stats);
  }

  if (stats.fresh_search_decisions == 0) {
    std::cerr << "WARNING: No fresh_search decisions found in " << trajectory_jsonl_path << "\n";
    return 1;
  }

  HeldOutReplayComputedMetrics metrics = ComputeHeldOutReplayMetrics(stats);
  PrintAndSaveHeldOutReplayReport(
      model_path, trajectory_jsonl_path, obs_bin_path, output_json_path, stats, metrics);
  return 0;
}

int RunTrueStateTieBreakerAndHistoryReplay() {
  const std::string games_jsonl_path = absl::GetFlag(FLAGS_tie_breaker_games_jsonl);
  const std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
  const std::string output_path = absl::GetFlag(FLAGS_output_json);
  const bool run_rollouts = absl::GetFlag(FLAGS_tie_breaker_run_rollouts);
  const int max_sample = absl::GetFlag(FLAGS_tie_breaker_max_sample);
  const int tie_breaker_rollouts = absl::GetFlag(FLAGS_tie_breaker_rollouts);
  const int parallel_roots = std::max(1, absl::GetFlag(FLAGS_parallel_roots));
  const int num_threads = std::max(1, absl::GetFlag(FLAGS_threads));
  const uint64_t master_seed = absl::GetFlag(FLAGS_master_seed);
  const int num_blocks = absl::GetFlag(FLAGS_num_blocks);
  const int num_evals = absl::GetFlag(FLAGS_num_evaluators);
  const int target_batch_size = absl::GetFlag(FLAGS_target_batch_size);

  if (games_jsonl_path.empty()) {
    open_spiel::SpielFatalError("--tie_breaker_games_jsonl must be specified.");
  }

  std::cout << "================================================================================\n";
  std::cout << "MODE 3: TRUE-STATE TIE-BREAKER & HISTORY REPLAY VERIFICATION\n";
  std::cout << "================================================================================\n";
  std::cout << "Games JSONL:        " << games_jsonl_path << "\n";
  std::cout << "Model checkpoint:   " << model_path << "\n";
  std::cout << "Output JSON:        " << output_path << "\n";
  std::cout << "Run rollouts:       " << (run_rollouts ? "true" : "false") << "\n";
  std::cout << "Max sample:         " << max_sample << "\n";
  std::cout << "Rollouts/action:    " << tie_breaker_rollouts << "\n";
  std::cout << "Threads:            " << num_threads << "\n";
  std::cout << "Parallel roots:     " << parallel_roots << "\n";

  open_spiel::dune_imperium::DuneImperiumState::SetAutoPassEmptyCombatIntrigueHands(
      absl::GetFlag(FLAGS_auto_pass_empty_combat_intrigue));

  at::globalContext().setAllowTF32CuBLAS(false);
  at::globalContext().setAllowTF32CuDNN(false);

  EvaluatorBundle bundle;
  if (!model_path.empty()) {
    bundle = InitEvaluatorBundle(
        num_evals, target_batch_size, num_blocks, model_path,
        /*rollout_amp=*/false, /*allow_tf32=*/false,
        /*disable_global_tf32=*/false, /*log_device=*/false);
  }
  auto& evaluators = bundle.evaluators;

  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("dune_imperium");
  std::vector<ParsedGame> parsed_games = LoadTieBreakerGames(games_jsonl_path);

  std::vector<std::vector<ChangedDecision>> per_game_changed(parsed_games.size());
  std::atomic<size_t> next_g_idx{0};
  std::vector<std::thread> workers;
  workers.reserve(num_threads);

  for (int t = 0; t < num_threads; ++t) {
    workers.emplace_back([&, t]() {
      int eval_idx = evaluators.empty() ? 0 : (t % evaluators.size());
      while (true) {
        size_t g = next_g_idx.fetch_add(1);
        if (g >= parsed_games.size()) break;
        const auto& pg = parsed_games[g];

        std::unique_ptr<State> state = game->NewInitialState();
        std::vector<int64_t> prefix;
        prefix.reserve(pg.history.size());

        for (size_t step = 0; step < pg.history.size(); ++step) {
          int64_t a = pg.history[step];
          if (!model_path.empty() && !state->IsChanceNode() && state->CurrentPlayer() == pg.seat) {
            auto legals = state->LegalActions();
            if (legals.size() > 1) {
              auto prior = evaluators[eval_idx]->Prior(*state);
              Action a_parent = legals.front();
              double max_p = -1.0;
              for (const auto& ap : prior) {
                if (ap.second > max_p) {
                  max_p = ap.second;
                  a_parent = ap.first;
                }
              }
              Action a_student = static_cast<Action>(a);
              if (a_student != a_parent) {
                ChangedDecision cd;
                cd.game_idx = static_cast<int>(g);
                cd.seat = pg.seat;
                cd.a_parent = a_parent;
                cd.a_student = a_student;
                cd.history_prefix = prefix;
                per_game_changed[g].push_back(std::move(cd));
              }
            }
          }
          state->ApplyAction(a);
          prefix.push_back(a);
        }

        if (!state->IsTerminal()) {
          open_spiel::SpielFatalError(absl::StrFormat("Game %d (episode %d) did not terminate at end of history!", g, pg.episode_id));
        }
        double r_ret = state->Returns()[pg.seat];
        if (std::abs(r_ret - pg.recorded_return) >= 1e-4) {
          open_spiel::SpielFatalError(absl::StrFormat("Game %d (episode %d) return mismatch: actual=%f, recorded=%f",
                                                     g, pg.episode_id, r_ret, pg.recorded_return));
        }
        int comp_placement = 4;
        if (std::abs(r_ret - 2.25) < 1e-4) comp_placement = 1;
        else if (std::abs(r_ret - 0.25) < 1e-4) comp_placement = 2;
        else if (std::abs(r_ret - (-0.75)) < 1e-4) comp_placement = 3;
        else if (std::abs(r_ret - (-1.75)) < 1e-4) comp_placement = 4;
        else {
          int rank = 1;
          for (int p = 0; p < 4; ++p) {
            if (p != pg.seat && state->Returns()[p] > r_ret + 1e-4) rank++;
          }
          comp_placement = rank;
        }
        if (comp_placement != pg.placement) {
          open_spiel::SpielFatalError(absl::StrFormat("Game %d (episode %d) placement mismatch: comp=%d, recorded=%d",
                                                     g, pg.episode_id, comp_placement, pg.placement));
        }
      }
    });
  }

  for (auto& w : workers) {
    w.join();
  }

  std::cout << "[VERIFY] 100% replay match verified (terminal, returns, and placements matched across all games).\n";

  std::vector<ChangedDecision> all_changed;
  for (size_t g = 0; g < parsed_games.size(); ++g) {
    for (auto& cd : per_game_changed[g]) {
      all_changed.push_back(std::move(cd));
    }
  }

  const size_t N_games = parsed_games.size();
  const size_t N_changed = all_changed.size();
  const double r = static_cast<double>(N_changed) / static_cast<double>(N_games);
  std::cout << absl::StrFormat("[TIE-BREAKER] Found %zu changed decisions across %zu games (r = %.4f changed/game)\n",
                               N_changed, N_games, r);

  bool has_rollout_stats = false;
  size_t sampled_count = 0;
  TieBreakerStats tb_stats;

  if (run_rollouts && N_changed > 0 && !evaluators.empty()) {
    size_t M = std::min<size_t>(static_cast<size_t>(max_sample), N_changed);
    std::mt19937 rng(master_seed);
    std::vector<size_t> perm(N_changed);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    perm.resize(M);
    std::sort(perm.begin(), perm.end());

    std::vector<ChangedDecision> sampled_decisions;
    sampled_decisions.reserve(M);
    for (size_t idx : perm) {
      sampled_decisions.push_back(all_changed[idx]);
    }

    std::cout << absl::StrFormat("[TIE-BREAKER] Running %d paired true-state rollouts on %zu sampled changed decisions...\n",
                                 tie_breaker_rollouts, M);

    std::vector<double> deltas(M, 0.0);
    std::atomic<size_t> next_sample_idx{0};
    std::vector<std::thread> roll_workers;
    roll_workers.reserve(parallel_roots);

    for (int pw = 0; pw < parallel_roots; ++pw) {
      roll_workers.emplace_back([&]() {
        while (true) {
          size_t i = next_sample_idx.fetch_add(1);
          if (i >= M) break;
          const auto& cd = sampled_decisions[i];
          std::unique_ptr<State> state = game->NewInitialState();
          for (int64_t a : cd.history_prefix) {
            state->ApplyAction(a);
          }
          uint64_t dec_seed = dune_seed::DeriveSeed(master_seed, 0x4509ULL, static_cast<uint64_t>(i));
          std::vector<Action> cands = {cd.a_parent, cd.a_student};
          auto s_res = PairedPriorSortedRolloutSearchDecision(
              *state, cd.seat, nullptr, evaluators, cands,
              tie_breaker_rollouts, /*min_override_margin=*/0.0, /*min_t_stat=*/0.0,
              dec_seed, SearchRolloutCrnMode::kSequential, /*sample_opponent_leader_draft=*/false,
              /*max_time_ms=*/-1.0, absl::GetFlag(FLAGS_max_inflight_worlds),
              /*resample_from_infostate=*/false, /*use_bounded_pool_when_unclocked=*/true);

          double u_student = 0.0, u_parent = 0.0;
          for (const auto& cu : s_res.candidate_utilities) {
            if (cu.first == cd.a_student) u_student = cu.second;
            if (cu.first == cd.a_parent) u_parent = cu.second;
          }
          deltas[i] = u_student - u_parent;

          if ((i + 1) % 20 == 0 || i + 1 == M) {
            std::cout << absl::StrFormat("  [TIE-BREAKER] Evaluated %zu / %zu changed moves (delta=%.4f)\n",
                                         i + 1, M, deltas[i]);
          }
        }
      });
    }

    for (auto& w : roll_workers) {
      w.join();
    }

    has_rollout_stats = true;
    sampled_count = M;
    tb_stats = ComputeTieBreakerBootstrapStats(
        N_games, master_seed, per_game_changed, sampled_decisions, deltas);
  }

  PrintAndSaveTieBreakerSummary(
      output_path, N_games, N_changed, r, has_rollout_stats,
      sampled_count, tie_breaker_rollouts, tb_stats);
  return 0;
}

}  // namespace open_spiel::dune_imperium::fixed_root_benchmark
