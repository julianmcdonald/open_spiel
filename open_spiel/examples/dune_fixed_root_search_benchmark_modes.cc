#include "open_spiel/examples/dune_fixed_root_search_benchmark_internal.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/examples/dune_seed_utils.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel::dune_imperium::fixed_root_benchmark {

int RunSelfPlayRootAndPreservationCapture() {
  const int num_games = absl::GetFlag(FLAGS_capture_self_play_games);
  const int roots_per_seat = absl::GetFlag(FLAGS_capture_roots_per_seat_game);
  const int pres_per_seat = absl::GetFlag(FLAGS_capture_pres_per_seat_game);
  const int max_placement_roots = absl::GetFlag(FLAGS_capture_max_placement_roots);
  const std::string out_dir = absl::GetFlag(FLAGS_capture_out_dir);
  const std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
  const int num_threads = std::max(1, absl::GetFlag(FLAGS_threads));
  const int num_evals = absl::GetFlag(FLAGS_num_evaluators);
  const int target_batch_size = absl::GetFlag(FLAGS_target_batch_size);
  const uint64_t base_seed = absl::GetFlag(FLAGS_master_seed);
  const int num_blocks = absl::GetFlag(FLAGS_num_blocks);

  if (out_dir.empty()) {
    open_spiel::SpielFatalError("--capture_out_dir must be specified when --capture_self_play_games > 0");
  }
  if (model_path.empty()) {
    open_spiel::SpielFatalError("--model_checkpoint must be specified when --capture_self_play_games > 0");
  }

  std::cout << "================================================================================\n";
  std::cout << "MODE 1: RAW GREEDY SELF-PLAY ROOT & PRESERVATION CAPTURE\n";
  std::cout << "================================================================================\n";
  std::cout << "Model checkpoint:   " << model_path << "\n";
  std::cout << "Output dir:         " << out_dir << "\n";
  std::cout << "Games:              " << num_games << "\n";
  std::cout << "Threads:            " << num_threads << "\n";
  std::cout << "Roots / seat-game:  " << roots_per_seat << "\n";
  std::cout << "Pres / seat-game:   " << pres_per_seat << "\n";
  std::cout << "Max placement roots:" << max_placement_roots << "\n";
  std::cout << "Master seed:        " << base_seed << "\n";
  std::cout << "Num blocks:         " << num_blocks << "\n";

  open_spiel::dune_imperium::DuneImperiumState::SetAutoPassEmptyCombatIntrigueHands(
      absl::GetFlag(FLAGS_auto_pass_empty_combat_intrigue));

  EvaluatorBundle bundle = InitEvaluatorBundle(
      num_evals, target_batch_size, num_blocks, model_path,
      /*rollout_amp=*/false, /*allow_tf32=*/false,
      /*disable_global_tf32=*/true, /*log_device=*/true);
  auto& evaluators = bundle.evaluators;

  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("dune_imperium");

  std::vector<std::vector<CapturedPlacementRoot>> per_game_roots(num_games);
  std::vector<std::vector<CapturedPreservationItem>> per_game_pres(num_games);

  std::filesystem::create_directories(out_dir);
  std::string ckpt_path = out_dir + "/preservation_pool.bin.ckpt";

  std::unique_ptr<std::atomic<uint8_t>[]> game_completed(new std::atomic<uint8_t>[num_games]());
  for (int i = 0; i < num_games; ++i) {
    game_completed[i].store(0, std::memory_order_relaxed);
  }
  std::atomic<int> completed_games_count{0};
  std::mutex ckpt_mutex;

  if (roots_per_seat == 0) {
    completed_games_count.store(
        LoadPreservationCheckpoint(ckpt_path, num_games, per_game_pres, game_completed.get()),
        std::memory_order_relaxed);
  }

  if (completed_games_count.load() == num_games) {
    std::cout << absl::StrFormat("[CAPTURE] All %d games already completed from checkpoint\n", num_games);
  }

  std::atomic<int> next_game{0};
  std::vector<std::thread> workers;
  workers.reserve(num_threads);

  auto start_time = std::chrono::steady_clock::now();

  for (int t = 0; t < num_threads; ++t) {
    workers.emplace_back([&, t]() {
      int eval_idx = t % num_evals;
      while (true) {
        int g = next_game.fetch_add(1);
        if (g >= num_games) break;
        if (game_completed[g].load(std::memory_order_acquire) != 0) continue;

        int mod20 = g % 20;
        std::string split = (mod20 < 14) ? "train" : (mod20 < 17 ? "val" : "test");
        int split_id = (mod20 < 14) ? 0 : (mod20 < 17 ? 1 : 2);

        uint64_t chance_seed = dune_seed::DeriveSeed(base_seed, 0x4501ULL, static_cast<uint64_t>(g));
        std::mt19937 rng(chance_seed);
        std::unique_ptr<State> state = game->NewInitialState();
        std::vector<int64_t> history;

        std::array<std::vector<CapturedPlacementRoot>, 4> seat_placements;
        std::array<std::vector<CapturedPreservationItem>, 4> seat_all_decisions;
        int decision_ordinal = 0;

        while (!state->IsTerminal()) {
          if (state->IsChanceNode()) {
            auto outcomes = state->ChanceOutcomes();
            double u = std::generate_canonical<double, 53>(rng);
            Action ca = open_spiel::SampleAction(outcomes, u).first;
            state->ApplyAction(ca);
            history.push_back(ca);
            continue;
          }

          Player cur = state->CurrentPlayer();
          std::vector<Action> legals = state->LegalActions();
          if (legals.empty()) break;

          if (legals.size() == 1) {
            Action a = legals.front();
            state->ApplyAction(a);
            history.push_back(a);
            continue;
          }

          const auto* dune = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
          GamePhase phase = dune ? dune->phase() : GamePhase::kAgentTurns;

          std::string dec_type;
          bool has_tech = false, has_tleilaxu = false, has_imperium = false;
          for (Action a : legals) {
            if ((70 <= a && a <= 75) || (770 <= a && a <= 775)) has_tech = true;
            if (90 <= a && a <= 95) has_tleilaxu = true;
            if ((10 <= a && a < 18) || (572 <= a && a < 580) || a == 598 || a == 599) has_imperium = true;
          }
          if (has_tech || has_tleilaxu || has_imperium) {
            if (has_tech) dec_type = "Purchase:Tech";
            else if (has_tleilaxu) dec_type = "Purchase:Tleilaxu";
            else if (has_imperium) dec_type = "Purchase:Imperium";
            else dec_type = "Purchase:Other";
          } else {
            bool has_leader = false;
            for (Action a : legals) {
              if (558 <= a && a <= 571) { has_leader = true; break; }
            }
            if (has_leader) {
              dec_type = "LeaderSelect";
            } else {
              bool has_combat_act = false;
              for (Action a : legals) {
                if (a == 40 || (1700 <= a && a < 1800) || (650 <= a && a < 698)) {
                  has_combat_act = true;
                  break;
                }
              }
              if (phase == GamePhase::kCombat || has_combat_act) {
                dec_type = "CombatIntrigue";
              } else if (phase == GamePhase::kAgentTurns) {
                bool has_card = false;
                bool has_space = false;
                for (Action a : legals) {
                  if ((1800 <= a && a < 2056) || a == 3) has_card = true;
                  if (600 <= a && a <= 621) has_space = true;
                }
                if (has_card && !has_space) dec_type = "AgentPrimary";
                else if (has_space) dec_type = "AgentContinuation";
                else dec_type = "Other";
              } else {
                dec_type = "Other";
              }
            }
          }

          auto prior = evaluators[eval_idx]->Prior(*state);
          Action raw_action = legals.front();
          double max_p = -1.0;
          std::vector<double> pi_p(legals.size(), 0.0);
          for (size_t i = 0; i < legals.size(); ++i) {
            Action a = legals[i];
            double p = 0.0;
            for (const auto& ap : prior) {
              if (ap.first == a) { p = ap.second; break; }
            }
            pi_p[i] = p;
            if (p > max_p) {
              max_p = p;
              raw_action = a;
            }
          }

          std::vector<float> obs = evaluators[eval_idx]->GetConsumedObservation(*state, cur);
          dune_semantic::CandidateActionData cand_data;
          if (dune) {
            dune_semantic::ExtractCandidateDescriptors(*dune, legals, &cand_data, dune_semantic::kDescriptorSchemaVersionV3);
          }
          int cur_round = dune ? dune->GetCurrentRound() : 1;
          std::string round_band = (cur_round <= 3) ? "R1-3" : (cur_round <= 6 ? "R4-6" : "R7+");

          CapturedPreservationItem pres_item;
          pres_item.game_id = g;
          pres_item.split_id = split_id;
          pres_item.player = cur;
          pres_item.obs = obs;
          pres_item.cand_data = cand_data;
          pres_item.legal_actions.assign(legals.begin(), legals.end());
          pres_item.pi_parent = pi_p;
          seat_all_decisions[cur].push_back(std::move(pres_item));

          if (roots_per_seat > 0 && (dec_type == "AgentPrimary" || dec_type == "AgentContinuation")) {
            CapturedPlacementRoot pl_root;
            pl_root.game_id = g;
            pl_root.decision_ordinal = decision_ordinal;
            pl_root.search_seat = cur;
            pl_root.round = cur_round;
            pl_root.round_band = round_band;
            pl_root.decision_type = dec_type;
            pl_root.legal_actions.assign(legals.begin(), legals.end());
            pl_root.raw_action = raw_action;
            pl_root.action_history = history;
            pl_root.split = split;
            pl_root.pi_parent = std::move(pi_p);
            pl_root.obs = std::move(obs);
            pl_root.cand_data = std::move(cand_data);
            seat_placements[cur].push_back(std::move(pl_root));
          }

          decision_ordinal++;
          state->ApplyAction(raw_action);
          history.push_back(raw_action);
        }

        for (int s = 0; s < 4; ++s) {
          auto& pl_list = seat_placements[s];
          if (!pl_list.empty()) {
            std::mt19937 seat_rng(dune_seed::DeriveSeed(base_seed, 0x4502ULL, static_cast<uint64_t>(g * 4 + s)));
            std::vector<size_t> p_idx(pl_list.size());
            std::iota(p_idx.begin(), p_idx.end(), 0);
            std::shuffle(p_idx.begin(), p_idx.end(), seat_rng);
            size_t k_take = std::min<size_t>(roots_per_seat, p_idx.size());
            p_idx.resize(k_take);
            std::sort(p_idx.begin(), p_idx.end());
            for (size_t idx : p_idx) {
              per_game_roots[g].push_back(std::move(pl_list[idx]));
            }
          }

          auto& all_list = seat_all_decisions[s];
          if (!all_list.empty()) {
            std::mt19937 pres_rng(dune_seed::DeriveSeed(base_seed, 0x4503ULL, static_cast<uint64_t>(g * 4 + s)));
            std::vector<size_t> a_idx(all_list.size());
            std::iota(a_idx.begin(), a_idx.end(), 0);
            std::shuffle(a_idx.begin(), a_idx.end(), pres_rng);
            size_t k_take = std::min<size_t>(pres_per_seat, a_idx.size());
            a_idx.resize(k_take);
            std::sort(a_idx.begin(), a_idx.end());
            for (size_t idx : a_idx) {
              per_game_pres[g].push_back(std::move(all_list[idx]));
            }
          }
        }

        game_completed[g].store(1, std::memory_order_release);
        int done = ++completed_games_count;

        if (done % 10 == 0 || done == num_games) {
          std::cout << absl::StrFormat("[CAPTURE] Completed %d / %d self-play games\n", done, num_games);
        }

        if (roots_per_seat == 0 && done % 200 == 0) {
          std::lock_guard<std::mutex> lock(ckpt_mutex);
          SavePreservationCheckpoint(out_dir, num_games, game_completed.get(), per_game_pres);
        }
      }
    });
  }

  for (auto& w : workers) {
    w.join();
  }

  std::vector<CapturedPlacementRoot> all_sampled_roots;
  for (int g = 0; g < num_games; ++g) {
    for (auto& r : per_game_roots[g]) {
      all_sampled_roots.push_back(std::move(r));
    }
  }

  if (max_placement_roots > 0 && all_sampled_roots.size() > static_cast<size_t>(max_placement_roots)) {
    std::mt19937 sub_rng(dune_seed::DeriveSeed(base_seed, 0x4504ULL, 0ULL));
    std::vector<size_t> indices(all_sampled_roots.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), sub_rng);
    indices.resize(max_placement_roots);
    std::sort(indices.begin(), indices.end());
    std::vector<CapturedPlacementRoot> trimmed;
    trimmed.reserve(max_placement_roots);
    for (size_t idx : indices) {
      trimmed.push_back(std::move(all_sampled_roots[idx]));
    }
    all_sampled_roots = std::move(trimmed);
  }

  std::vector<CapturedPreservationItem> all_pres;
  for (int g = 0; g < num_games; ++g) {
    for (auto& p : per_game_pres[g]) {
      all_pres.push_back(std::move(p));
    }
  }

  SaveCaptureOutputs(*game, out_dir, num_games, base_seed, all_sampled_roots, all_pres, 0.0);

  auto end_time = std::chrono::steady_clock::now();
  double elapsed_s = std::chrono::duration<double>(end_time - start_time).count();
  std::cout << absl::StrFormat("[CAPTURE] Complete in %.1f s. Sampled %zu placement roots, %zu preservation decisions.\n",
                               elapsed_s, all_sampled_roots.size(), all_pres.size());
  return 0;
}

int RunSuccessiveHalvingDistillLabel() {
  const std::string model_path = absl::GetFlag(FLAGS_model_checkpoint);
  const std::string roots_path = absl::GetFlag(FLAGS_sampled_roots_json);
  const std::string obs_path = absl::GetFlag(FLAGS_obs_binary);
  const std::string output_path = absl::GetFlag(FLAGS_output_json);
  const int parallel_roots = std::max(1, absl::GetFlag(FLAGS_parallel_roots));
  const int num_evals = absl::GetFlag(FLAGS_num_evaluators);
  const int target_batch_size = absl::GetFlag(FLAGS_target_batch_size);
  const uint64_t master_seed = absl::GetFlag(FLAGS_master_seed);
  const uint64_t label_seed_domain = absl::GetFlag(FLAGS_label_seed_domain);
  const bool use_bounded_pool = absl::GetFlag(FLAGS_use_bounded_pool_when_unclocked);
  const int max_inflight = absl::GetFlag(FLAGS_max_inflight_worlds);
  const bool placement_only = absl::GetFlag(FLAGS_placement_roots_only);
  const int num_blocks = absl::GetFlag(FLAGS_num_blocks);

  if (model_path.empty() || roots_path.empty() || obs_path.empty() || output_path.empty()) {
    open_spiel::SpielFatalError("--model_checkpoint, --sampled_roots_json, --obs_binary, and --output_json must be specified.");
  }

  std::cout << "================================================================================\n";
  std::cout << "MODE 2: PARALLEL-ROOTS SH_768_K5_ALLBUY LABELLING MODE\n";
  std::cout << "================================================================================\n";
  std::cout << "Model checkpoint:   " << model_path << "\n";
  std::cout << "Sampled roots JSON: " << roots_path << "\n";
  std::cout << "Observation bin:    " << obs_path << "\n";
  std::cout << "Output JSON:        " << output_path << "\n";
  std::cout << "Parallel roots:     " << parallel_roots << "\n";
  std::cout << "Master seed:        " << master_seed << "\n";
  std::cout << "Label seed domain:  " << label_seed_domain << "\n";
  std::cout << "Placement only:     " << (placement_only ? "true" : "false") << "\n";

  if (absl::GetFlag(FLAGS_sh_budget_check)) {
    open_spiel::dune_imperium::DuneImperiumState::SetAutoPassEmptyCombatIntrigueHands(false);
  } else {
    open_spiel::dune_imperium::DuneImperiumState::SetAutoPassEmptyCombatIntrigueHands(
        absl::GetFlag(FLAGS_auto_pass_empty_combat_intrigue));
  }

  EvaluatorBundle bundle = InitEvaluatorBundle(
      num_evals, target_batch_size, num_blocks, model_path,
      /*rollout_amp=*/false, /*allow_tf32=*/false,
      /*disable_global_tf32=*/true, /*log_device=*/true);
  auto& evaluators = bundle.evaluators;

  std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("dune_imperium");

  RootLoadVerifyOptions root_opts;
  root_opts.verify_game_string = false;
  root_opts.filter_placement_only = placement_only;
  root_opts.check_legal_during_replay = false;
  root_opts.check_obs_stream_good = false;
  root_opts.fill_missing_pi_parent = true;
  root_opts.require_non_empty = false;
  root_opts.log_loaded_count = false;
  root_opts.verbose_parity_messages = false;

  std::vector<SampledRoot> sampled_roots = LoadSampledRootsJson(roots_path, "", root_opts);
  VerifySampledRootsObsParity(*game, obs_path, evaluators[0], root_opts, sampled_roots);

  std::vector<LabelledResult> results(sampled_roots.size());
  std::unique_ptr<std::atomic<uint8_t>[]> completed(new std::atomic<uint8_t>[sampled_roots.size()]());
  std::string ckpt_path = output_path + ".checkpoint.json";
  std::atomic<int> completed_count{
      LoadLabelCheckpoint(ckpt_path, sampled_roots, results, completed.get())};

  std::mutex save_mutex;
  std::atomic<size_t> next_root_idx{0};
  std::vector<std::thread> workers;
  workers.reserve(parallel_roots);

  auto search_start = std::chrono::steady_clock::now();

  for (int w = 0; w < parallel_roots; ++w) {
    workers.emplace_back([&]() {
      while (true) {
        size_t r_idx = next_root_idx.fetch_add(1);
        if (r_idx >= sampled_roots.size()) break;
        if (completed[r_idx].load(std::memory_order_acquire) != 0) continue;

        const auto& root = sampled_roots[r_idx];
        std::unique_ptr<State> state = game->NewInitialState();
        for (int64_t a : root.action_history) {
          state->ApplyAction(a);
        }

        uint64_t dec_seed = dune_seed::DeriveSeed(master_seed, label_seed_domain, static_cast<uint64_t>(root.root_id));
        int eff_top_k = (root.decision_type.rfind("Purchase", 0) == 0)
            ? static_cast<int>(root.legal_actions.size()) : 5;

        auto t0 = std::chrono::steady_clock::now();
        auto s_res = SuccessiveHalvingRolloutSearchDecision(
            *state, root.search_seat, nullptr, evaluators,
            /*reference_rollouts_per_action=*/256, /*min_override_margin=*/0.15,
            dec_seed, SearchRolloutCrnMode::kSequential, /*unbiased_stage2_gate=*/false,
            /*max_time_ms=*/-1.0, eff_top_k, max_inflight, use_bounded_pool);
        auto t1 = std::chrono::steady_clock::now();
        double dur = std::chrono::duration<double, std::milli>(t1 - t0).count();

        LabelledResult lr;
        lr.root_id = root.root_id;
        lr.game_id = root.game_id;
        lr.obs_row = root.obs_row;
        lr.search_seat = root.search_seat;
        lr.round = root.round;
        lr.decision_type = root.decision_type;
        lr.split = root.split;
        lr.raw_action = root.raw_action;
        lr.chosen_action = s_res.action;
        lr.overridden = s_res.overridden;
        lr.d2 = s_res.mean_d_stage2;
        lr.se2 = s_res.se_d_stage2;
        lr.w = s_res.overridden ? std::clamp((lr.d2 - lr.se2) / 0.15, 0.0, 1.0) : 0.0;
        for (const auto& cu : s_res.candidate_utilities) {
          lr.candidates.push_back(cu.first);
        }
        lr.legal_actions = root.legal_actions;
        lr.pi_parent = root.pi_parent;
        lr.duration_ms = dur;
        lr.stage1_kept_worlds = s_res.stage1_kept_worlds;
        lr.stage2_kept_worlds = s_res.stage2_kept_worlds;

        results[r_idx] = std::move(lr);
        completed[r_idx].store(1, std::memory_order_release);
        int done = completed_count.fetch_add(1) + 1;

        if (done % 10 == 0 || done == static_cast<int>(sampled_roots.size())) {
          std::cout << absl::StrFormat("[LABEL] Completed %d / %zu roots (Root %d, over=%s, w=%.3f, dur=%.1f ms)\n",
                                       done, sampled_roots.size(), root.root_id,
                                       results[r_idx].overridden ? "true" : "false", results[r_idx].w, dur);
        }
        if (done % 20 == 0 || done == static_cast<int>(sampled_roots.size())) {
          std::lock_guard<std::mutex> lock(save_mutex);
          SaveDistillLabelCheckpoint(ckpt_path, sampled_roots.size(), completed_count.load(), results, completed.get());
        }
      }
    });
  }

  for (auto& w : workers) {
    w.join();
  }

  auto search_end = std::chrono::steady_clock::now();
  double total_wall_s = std::chrono::duration<double>(search_end - search_start).count();
  DistillLabelStats stats = ComputeDistillLabelStats(results, total_wall_s);
  SaveDistillLabelFinalOutput(output_path, ckpt_path, results, stats);
  return 0;
}

}  // namespace open_spiel::dune_imperium::fixed_root_benchmark
