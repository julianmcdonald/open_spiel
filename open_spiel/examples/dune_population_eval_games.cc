// Game simulation and worker pool for dune_population_eval.

#include "dune_population_eval_internal.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"

#include "dune_compound_turn_search.h"
#include "dune_eval_action_selection.h"
#include "dune_seed_utils.h"
#include "dune_specimen_conversion.h"

namespace open_spiel {
namespace dune_pop_eval {

std::atomic<int64_t> g_teacher_searched_roots{0};
std::atomic<int64_t> g_teacher_overrides{0};

// ---------------------------------------------------------------------------
// Worker thread: plays games with deterministic per-game seeds.
// Results are stored in a pre-allocated vector indexed by episode_id,
// guaranteeing thread-count-independent output.
// ---------------------------------------------------------------------------
void WorkerThread(
    int /*thread_id*/,
    std::shared_ptr<const Game> game,
    std::shared_ptr<IGameEvaluator> model_evaluator,
    const std::vector<std::shared_ptr<IGameEvaluator>>& opponent_evaluators,
    const std::vector<std::string>& opponent_names,
    int64_t obs_size,
    int candidate_input_dim,
    dune_imperium::MarketAppendixMode candidate_market_mode,
    const std::vector<OpponentMetadata>& opp_metadata,
    bool provides_info_state_tensor,
    bool provides_observations_tensor,
    std::atomic<int>& next_game_id,
    int start_episode_id,
    int total_games,
    uint64_t base_seed,
    uint64_t domain,
    bool greedy,
    float temperature,
    // PWO-5 gate 2 item (d). `has_*` false = inherit the global value above.
    bool has_opponent_greedy,
    bool opponent_greedy,
    bool has_opponent_temperature,
    float opponent_temperature,
    float candidate_logit_cap,
    float opponent_logit_cap,
    std::ofstream* dump_out,
    std::vector<GameResult>& results,
    bool record_market_diagnostics,
    std::vector<dune_market_diag::CompactPerGameMarketDiagnostics>* market_results,
    bool record_history) {
  // Pre-allocate observation buffers
  std::vector<float> cand_obs(candidate_input_dim, 0.0f);
  std::vector<std::vector<float>> opp_obs;
  opp_obs.reserve(opponent_evaluators.size());
  for (size_t i = 0; i < opponent_evaluators.size(); ++i) {
    int dim = (i < opp_metadata.size()) ? opp_metadata[i].input_dim : static_cast<int>(obs_size);
    opp_obs.emplace_back(dim, 0.0f);
  }

  std::vector<std::shared_ptr<open_spiel::BatchedNNEvaluator>> teacher_evals;
  open_spiel::dune_imperium::SearchRolloutCrnMode teacher_crn_mode =
      open_spiel::dune_imperium::SearchRolloutCrnMode::kDeckOrderCrn;
  if (absl::GetFlag(FLAGS_candidate_r758_teacher)) {
    auto batched_cand = std::dynamic_pointer_cast<open_spiel::BatchedEvaluator>(model_evaluator);
    if (!batched_cand) {
      SpielFatalError("Candidate teacher search requires BatchedEvaluator (--deterministic_eval=false)");
    }
    teacher_evals.push_back(std::make_shared<open_spiel::BatchedNNEvaluator>(
        batched_cand, candidate_logit_cap, candidate_market_mode,
        /*forward_active_rollouts=*/false));
    const std::string mode_str = absl::GetFlag(FLAGS_teacher_crn_mode);
    if (mode_str == "sequential") {
      teacher_crn_mode = open_spiel::dune_imperium::SearchRolloutCrnMode::kSequential;
    } else if (mode_str == "deck_order") {
      teacher_crn_mode = open_spiel::dune_imperium::SearchRolloutCrnMode::kDeckOrderCrn;
    } else if (mode_str == "event_keyed") {
      teacher_crn_mode = open_spiel::dune_imperium::SearchRolloutCrnMode::kEventKeyedCrn;
    } else {
      SpielFatalError(absl::StrCat("Unknown teacher_crn_mode: ", mode_str));
    }
  }

  while (true) {
    const int result_index = next_game_id++;
    if (result_index >= total_games) break;
    const int episode_id = start_episode_id + result_index;

    // Log progress every 100 completed games
    int completed = next_game_id.load();
    if (completed % 100 == 0) {
      static std::mutex progress_mutex;
      static int last_printed = -1;
      std::lock_guard<std::mutex> lock(progress_mutex);
      if (completed > last_printed) {
        std::cout << "Progress: " << completed << " / " << total_games << " games completed..." << std::endl;
        last_printed = completed;
      }
    }

    // --- Round-robin seat assignment for exact balance ---
    int model_player = episode_id % kNumPlayers;

    // --- Per-game chance RNG ---
    uint64_t chance_seed = dune_seed::DeriveSeed(
        base_seed, domain, static_cast<uint64_t>(episode_id),
        dune_seed::kStreamChance);
    std::mt19937 chance_rng = dune_seed::MakeRng32(chance_seed);

    // --- Per-player policy RNGs (for stochastic mode) ---
    std::array<std::mt19937_64, 4> policy_rngs;
    for (int p = 0; p < kNumPlayers; ++p) {
      uint64_t policy_stream = dune_seed::kStreamPolicyPlayer0 +
                               static_cast<uint64_t>(p);
      uint64_t pseed = dune_seed::DeriveSeed(
          base_seed, domain, static_cast<uint64_t>(episode_id),
          policy_stream);
      policy_rngs[p] = dune_seed::MakeRng64(pseed);
    }

    // --- Per-game opponent assignment RNG ---
    std::vector<size_t> player_opp_idx(kNumPlayers, 0);
    if (!opponent_evaluators.empty()) {
      uint64_t opp_assign_seed = dune_seed::DeriveSeed(
          base_seed, domain, static_cast<uint64_t>(episode_id),
          dune_seed::kStreamOpponentAssign);
      std::mt19937_64 opp_rng = dune_seed::MakeRng64(opp_assign_seed);
      std::uniform_int_distribution<size_t> opp_dist(0, opponent_evaluators.size() - 1);

      for (int p = 0; p < kNumPlayers; ++p) {
        if (p == model_player) continue;
        player_opp_idx[p] = opp_dist(opp_rng);
      }
    }

    std::unique_ptr<State> state = game->NewInitialState();
    int game_length = 0;
    bool cand_space_placed_this_turn = false;
    Player last_turn_player = kInvalidPlayer;
    Action last_action = kInvalidAction;
    int64_t decision_index = 0;

    dune_market_diag::GameMarketTracker market_tracker;
    if (record_market_diagnostics) {
      market_tracker.InitGame(episode_id, model_player);
    }

    // Phase-3: observe running VP at each round boundary via public accessors.
    const DuneImperiumState* dune_state =
        dynamic_cast<const DuneImperiumState*>(state.get());
    std::array<std::array<int, kNumPlayers>, kMaxRounds + 2> vp_at_round_end{};
    std::array<std::array<int, kNumPlayers>, kMaxRounds + 2> solari_at_round_end{};
    std::array<int, kNumPlayers> reveals_with_agents_remaining{};
    std::array<bool, kMaxRounds + 2> round_end_seen{};
    std::array<int, kNumPlayers> specimen_conversions{};
    std::array<bool, kNumPlayers> atomics_used{};
    std::array<int, kNumPlayers> atomics_in_reveal{-1, -1, -1, -1};
    std::array<int, kNumPlayers> atomics_persuasion{-1, -1, -1, -1};
    std::array<int, kNumPlayers> atomics_round{-1, -1, -1, -1};
    std::array<int, kNumPlayers> plot_plays{};
    std::array<std::vector<int>, kNumPlayers> plot_pre_hand_sizes{};
    std::array<std::vector<std::pair<int, int>>, kNumPlayers> game_research_paths{};
    for (int p = 0; p < kNumPlayers; ++p) {
      int c = dune_state ? dune_state->GetResearchBottomColForTesting(p) : 0;
      int r = dune_state ? dune_state->GetResearchBottomRowForTesting(p) : 0;
      game_research_paths[p].push_back({c, r});
    }
    int last_round = dune_state ? dune_state->GetCurrentRound() : -1;
    const int trace_ep = absl::GetFlag(FLAGS_trace_episode);
    const bool trace_this = (trace_ep >= 0 && episode_id == trace_ep) || (trace_ep == -2);

    if (trace_this) {
      std::cout << "\n======================================================\n";
      std::cout << ">>> START TRACE FOR EPISODE " << episode_id << " <<<\n";
      std::cout << "Candidate Seat: " << model_player << "\n";
      std::cout << "Leaders: ";
      if (dune_state) {
        for (int p = 0; p < kNumPlayers; ++p) {
          std::cout << "P" << p << "=" << dune_state->PlayerLeader(p) << " ";
        }
      }
      std::cout << "\n======================================================\n";
    }

    auto snapshot_round_ends = [&]() {
      if (!dune_state) return;
      int cur = dune_state->GetCurrentRound();
      const int limit = state->IsTerminal() ? (cur + 1) : cur;
      while (last_round >= 0 && last_round < limit) {
        if (last_round >= 1 && last_round <= kMaxRounds) {
          for (int p = 0; p < kNumPlayers; ++p) {
            vp_at_round_end[last_round][p] = dune_state->GetPlayerVp(p);
            solari_at_round_end[last_round][p] = dune_state->GetPlayerSolariForTesting(p);
          }
          round_end_seen[last_round] = true;
          if (trace_this) {
            std::cout << "\n--- [ROUND " << last_round << " COMPLETED] ---\n";
            for (int p = 0; p < kNumPlayers; ++p) {
              std::cout << "  P" << p << (p == model_player ? " (CANDIDATE)" : "            ")
                        << ": VP=" << dune_state->GetPlayerVp(p)
                        << " | Solari=" << dune_state->GetPlayerSolariForTesting(p)
                        << " Spice=" << dune_state->GetPlayerSpiceForTesting(p)
                        << " Water=" << dune_state->GetPlayerWaterForTesting(p)
                        << " Garrison=" << dune_state->GetPlayerTroopsForTesting(p)
                        << " Combat=" << dune_state->TroopsInCombat(p)
                        << " | Influence: [Emp=" << dune_state->GetPlayerInfluenceForTesting(p, dune_imperium::Faction::kEmperor)
                        << " Guild=" << dune_state->GetPlayerInfluenceForTesting(p, dune_imperium::Faction::kSpacingGuild)
                        << " BG=" << dune_state->GetPlayerInfluenceForTesting(p, dune_imperium::Faction::kBeneGesserit)
                        << " Frem=" << dune_state->GetPlayerInfluenceForTesting(p, dune_imperium::Faction::kFremen) << "]"
                        << " | Alliances: [";
              for (int f = 0; f < 4; ++f) {
                if (dune_state->GetAllianceOwnerForTesting(f) == p) std::cout << " F" << f;
              }
              std::cout << " ]\n";
            }
            std::cout << "------------------------------------\n\n";
          }
        }
        ++last_round;
      }
    };

    while (!state->IsTerminal()) {
      snapshot_round_ends();
      ++game_length;
      if (game_length > 5000) {
        std::cerr << "Possible infinite loop in episode " << episode_id
                  << "! Length: " << game_length
                  << " Player: " << state->CurrentPlayer() << std::endl;
        std::abort();
      }

      if (state->IsChanceNode()) {
        auto outcomes = state->ChanceOutcomes();
        Action action;
        if (game->GetType().chance_mode ==
            GameType::ChanceMode::kSampledStochastic) {
          action = outcomes.front().first;
        } else {
          action = SampleAction(outcomes, chance_rng).first;
        }
        if (trace_this) {
          std::cout << "[R" << (dune_state ? dune_state->GetCurrentRound() : -1)
                    << " CHANCE] " << state->ActionToString(kChancePlayerId, action) << "\n";
        }
        state->ApplyAction(action);
        continue;
      }

      Player current_player = state->CurrentPlayer();
      std::vector<Action> legal_actions = state->LegalActions();
      if (legal_actions.empty()) {
        std::cerr << "Empty legal actions in episode " << episode_id << "!\n";
        break;
      }

      ++decision_index;
      if ((current_player != last_turn_player || last_action == dune_imperium::kActionEndTurn) &&
          current_player == model_player) {
        cand_space_placed_this_turn = false;
      }

      if (record_market_diagnostics && dune_state != nullptr && current_player == model_player) {
        market_tracker.OnCandidateDecision(*dune_state, model_player, legal_actions);
      }

      Action chosen_action = -1;
      bool use_model = (current_player == model_player);
      bool use_opponent_model = (!use_model && !opponent_evaluators.empty());

      if (use_model || use_opponent_model) {
        std::shared_ptr<IGameEvaluator> evaluator;
        std::vector<float>* p_obs = nullptr;
        if (use_model) {
          evaluator = model_evaluator;
          p_obs = &cand_obs;
          std::fill(cand_obs.begin(), cand_obs.end(), 0.0f);
          if (dune_state != nullptr) {
            dune_state->InformationStateTensorWithAppendix(
                current_player, evaluator->MarketMode(), absl::MakeSpan(cand_obs),
                evaluator->DeckPoolSemantics());
          } else if (provides_info_state_tensor) {
            state->InformationStateTensor(current_player, absl::MakeSpan(cand_obs));
          } else if (provides_observations_tensor) {
            state->ObservationTensor(current_player, absl::MakeSpan(cand_obs));
          }
        } else {
          size_t idx = player_opp_idx[current_player];
          evaluator = opponent_evaluators[idx];
          p_obs = &opp_obs[idx];
          std::fill(opp_obs[idx].begin(), opp_obs[idx].end(), 0.0f);
          if (dune_state != nullptr) {
            dune_state->InformationStateTensorWithAppendix(
                current_player, evaluator->MarketMode(), absl::MakeSpan(opp_obs[idx]),
                evaluator->DeckPoolSemantics());
          } else if (provides_info_state_tensor) {
            state->InformationStateTensor(current_player, absl::MakeSpan(opp_obs[idx]));
          } else if (provides_observations_tensor) {
            state->ObservationTensor(current_player, absl::MakeSpan(opp_obs[idx]));
          }
        }

        dune_semantic::CandidateActionData cand_data;
        const dune_semantic::CandidateActionData* p_cand_data = nullptr;
        if (dune_state != nullptr) {
          const std::string schema = evaluator ? evaluator->SemanticDescriptorSchema() : dune_semantic::kDescriptorSchemaVersionV3;
          dune_semantic::ExtractCandidateDescriptors(*dune_state, legal_actions, &cand_data, schema);
          p_cand_data = &cand_data;
        }
        EvalResult result = evaluator->EvaluateWithActions(*p_obs, p_cand_data);
        if (use_model && dump_out != nullptr && dune_state != nullptr) {
          AppendLogitStatsRow(*dump_out, episode_id,
                              dune_state->GetCurrentRound(),
                              GamePhaseRole(dune_state->phase()),
                              legal_actions, result.logits,
                              static_cast<double>(candidate_logit_cap));
        }
        const float logit_cap =
            use_model ? candidate_logit_cap : opponent_logit_cap;
        CenterAndCapLegalLogits(result.logits, legal_actions, logit_cap);

        // PWO-5 gate 2 item (d): the candidate seat and the opponent seats can
        // now select differently. With the overrides unset this resolves to the
        // global (greedy, temperature) for BOTH, which is the pre-gate-2
        // behaviour bit for bit -- see dune_eval_action_selection.h.
        const dune_eval::SelectionPolicy policy =
            dune_eval::ResolveSelectionPolicy(
                use_model, greedy, temperature, has_opponent_greedy,
                opponent_greedy, has_opponent_temperature,
                opponent_temperature);
        chosen_action = dune_eval::SelectActionFromLogits(
            result.logits, legal_actions, policy,
            policy_rngs[current_player]);

        if (use_model && absl::GetFlag(FLAGS_candidate_r758_teacher)) {
          auto cat = dune_imperium::ClassifyR755SearchCategory(
              *state, current_player, cand_space_placed_this_turn,
              absl::GetFlag(FLAGS_teacher_category_mask));
          if (cat != dune_imperium::R755SearchCategory::kIneligible) {
            double max_logit = -std::numeric_limits<double>::infinity();
            for (Action a : legal_actions) {
              if (static_cast<size_t>(a) < result.logits.size()) {
                max_logit = std::max(max_logit, static_cast<double>(result.logits[a]));
              }
            }
            double sum_exp = 0.0;
            std::vector<double> exps(legal_actions.size(), 0.0);
            for (size_t i = 0; i < legal_actions.size(); ++i) {
              Action a = legal_actions[i];
              if (static_cast<size_t>(a) < result.logits.size()) {
                double e = std::exp(static_cast<double>(result.logits[a]) - max_logit);
                exps[i] = e;
                sum_exp += e;
              }
            }
            open_spiel::ActionsAndProbs precomputed_prior;
            precomputed_prior.reserve(legal_actions.size());
            for (size_t i = 0; i < legal_actions.size(); ++i) {
              double p = (sum_exp > 0.0) ? (exps[i] / sum_exp) : (1.0 / legal_actions.size());
              precomputed_prior.push_back({legal_actions[i], p});
            }

            dune_imperium::CompoundSearchOptions opts;
            opts.precomputed_raw_prior = &precomputed_prior;
            opts.dedup_paired_worlds = true;
            opts.stage1_rollouts = absl::GetFlag(FLAGS_teacher_stage1_rollouts);
            opts.stage1_min_mean_diff = absl::GetFlag(FLAGS_teacher_stage1_min_mean_diff);

            uint64_t search_seed = dune_seed::DeriveSeed(
                base_seed, domain, static_cast<uint64_t>(episode_id),
                static_cast<uint64_t>(decision_index), 0x52373553ULL);

            dune_imperium::TurnSearchTree local_tree;
            auto sres = dune_imperium::CompoundRolloutSearchDecision(
                *state, current_player, &local_tree, teacher_evals,
                absl::GetFlag(FLAGS_teacher_top_k),
                absl::GetFlag(FLAGS_teacher_rollouts),
                absl::GetFlag(FLAGS_teacher_override_margin),
                dune_imperium::TreeReuseMode::kTargetFill,
                search_seed,
                dune_imperium::SearchPairingMode::kPaired,
                dune_imperium::SearchTruncationMode::kNextRoundSearchPlayer,
                teacher_crn_mode,
                /*world_sampler=*/nullptr,
                absl::GetFlag(FLAGS_teacher_min_t_stat),
                opts);

            ++g_teacher_searched_roots;
            if (sres.overridden) {
              ++g_teacher_overrides;
              chosen_action = sres.chosen_action;
            }
          }
        }
      } else {
        // Random opponent (no model loaded)
        std::uniform_int_distribution<size_t> dist(
            0, legal_actions.size() - 1);
        chosen_action = legal_actions[dist(policy_rngs[current_player])];
      }

      // WO-1 Phase 3 addendum: count applied specimen->troop conversions per
      // seat, at application time, by engine action-ID range. Measurement only
      // -- no shaping is enabled anywhere by this counter.
      //
      // PWO-5 gate 2 item (c): the range is 741-752, not 740-752. This site was
      // NOT named in the registration's known list (which named the trainer's
      // guard and dune_search_benchmark's counter); the registered "sweep for
      // other 740-752 sites" found it. Numerically a no-op -- 740 is an unused
      // base constant and is never legal -- so the committed PWO-2 popeval
      // conversion baselines (4.730 / 5.232 / 4.165 candidate-seat) do not move.
      if (dune_shaping::IsSpecimenConversionAction(chosen_action) &&
          current_player >= 0 && current_player < kNumPlayers) {
        ++specimen_conversions[current_player];
      }
      if (chosen_action == dune_imperium::kActionFamilyAtomics &&
          current_player >= 0 && current_player < kNumPlayers && dune_state != nullptr) {
        atomics_used[current_player] = true;
        atomics_round[current_player] = dune_state->GetCurrentRound();
        atomics_in_reveal[current_player] = dune_state->IsPlayerInRevealTurn(current_player) ? 1 : 0;
        atomics_persuasion[current_player] = dune_state->PlayerPersuasion(current_player);
      }
      if (chosen_action >= 1600 && chosen_action <= 1699 &&
          current_player >= 0 && current_player < kNumPlayers && dune_state != nullptr) {
        ++plot_plays[current_player];
        plot_pre_hand_sizes[current_player].push_back(dune_state->GetIntrigueHandForTesting(current_player).size());
      }
      if (chosen_action == dune_imperium::kActionReveal &&
          current_player >= 0 && current_player < kNumPlayers && dune_state != nullptr) {
        if (dune_state->GetPlayerAgentsRemainingForTesting(current_player) > 0) {
          ++reveals_with_agents_remaining[current_player];
        }
      }
      if (record_market_diagnostics && dune_state != nullptr) {
        market_tracker.BeforeApplyAction(*dune_state, current_player, model_player, chosen_action);
      }
      int prev_vp[kNumPlayers];
      for (int p = 0; p < kNumPlayers; ++p) {
        prev_vp[p] = dune_state ? dune_state->GetPlayerVp(p) : 0;
      }
      if (current_player == model_player &&
          chosen_action >= dune_imperium::kActionAgentSpaceConspire &&
          chosen_action <= dune_imperium::kActionAgentSpaceSietchTabr) {
        cand_space_placed_this_turn = true;
      }
      last_turn_player = current_player;
      last_action = chosen_action;
      state->ApplyAction(chosen_action);
      if (trace_this) {
        std::cout << "[R" << (dune_state ? dune_state->GetCurrentRound() : -1)
                  << " P" << current_player << (current_player == model_player ? " (CANDIDATE)" : "") << "] "
                  << state->ActionToString(current_player, chosen_action) << "\n";
        if (dune_state) {
          for (int p = 0; p < kNumPlayers; ++p) {
            int cur_v = dune_state->GetPlayerVp(p);
            if (cur_v != prev_vp[p]) {
              std::cout << "   >>> VP CHANGE for P" << p << (p == model_player ? " (CANDIDATE)" : "")
                        << ": " << prev_vp[p] << " -> " << cur_v
                        << " (delta " << (cur_v >= prev_vp[p] ? "+" : "") << (cur_v - prev_vp[p]) << ")\n";
              const auto& events = dune_state->GetVpEvents(p);
              if (!events.empty()) {
                const auto& ev = events.back();
                std::cout << "       Source: " << dune_imperium::VpSourceName(ev.source)
                          << " (Round " << ev.round << ", delta " << static_cast<int>(ev.delta) << ")\n";
              }
            }
          }
        }
      }
      if (dune_state != nullptr) {
        for (int p = 0; p < kNumPlayers; ++p) {
          int cur_c = dune_state->GetResearchBottomColForTesting(p);
          int cur_r = dune_state->GetResearchBottomRowForTesting(p);
          if (cur_c != game_research_paths[p].back().first ||
              cur_r != game_research_paths[p].back().second) {
            game_research_paths[p].push_back({cur_c, cur_r});
          }
        }
      }
      if (record_market_diagnostics && dune_state != nullptr) {
        market_tracker.AfterApplyAction(*dune_state, current_player, model_player, chosen_action);
      }
    }

    // Phase-3: capture the final (terminal) round's VP snapshot.
    snapshot_round_ends();

    if (trace_this && dune_state) {
      std::cout << "\n======================================================\n";
      std::cout << ">>> FINAL GAME STATE & VP ATTRIBUTION FOR EPISODE " << episode_id << " <<<\n";
      std::cout << "======================================================\n";
      for (int p = 0; p < kNumPlayers; ++p) {
        std::cout << "\n--- PLAYER " << p << (p == model_player ? " (CANDIDATE)" : " (OPPONENT)")
                  << " | Leader: " << dune_state->PlayerLeader(p)
                  << " | FinalScoredVp: " << dune_state->FinalScoredVp(p)
                  << " | TrackVp: " << dune_state->GetPlayerVp(p) << " ---\n";
        const auto& evs = dune_state->GetVpEvents(p);
        int run_vp = 0;
        for (size_t i = 0; i < evs.size(); ++i) {
          run_vp += evs[i].delta;
          std::cout << "  Event #" << (i + 1)
                    << ": Round " << evs[i].round
                    << " | Delta: " << (evs[i].delta >= 0 ? "+" : "") << static_cast<int>(evs[i].delta)
                    << " | Total: " << run_vp
                    << " | Source: " << dune_imperium::VpSourceName(evs[i].source) << "\n";
        }
      }
      std::cout << "======================================================\n\n";
    }

    // --- Collect results ---
    std::vector<double> returns = state->Returns();

    // Map returns cleanly directly to placement:
    // 2.25 -> 1st, 0.25 -> 2nd, -0.75 -> 3rd, -1.75 -> 4th
    int placement = 4;
    double r = returns[model_player];
    if (std::abs(r - 2.25) < 1e-4) {
      placement = 1;
    } else if (std::abs(r - 0.25) < 1e-4) {
      placement = 2;
    } else if (std::abs(r - (-0.75)) < 1e-4) {
      placement = 3;
    } else if (std::abs(r - (-1.75)) < 1e-4) {
      placement = 4;
    }

    GameResult& gr = results[result_index];
    gr.episode_id = episode_id;
    gr.seat = model_player;
    if (record_history) {
      gr.history = state->History();
    }

    // Track opponent names for seats other than model_player
    gr.opponents.clear();
    for (int p = 0; p < kNumPlayers; ++p) {
      if (p == model_player) continue;
      if (opponent_names.empty()) {
        gr.opponents.push_back("random");
      } else {
        size_t opp_idx = player_opp_idx[p];
        gr.opponents.push_back(opponent_names[opp_idx]);
      }
    }

    gr.placement = placement;
    gr.chance_seed = chance_seed;
    gr.game_return = returns[model_player];
    if (dune_state) {
      gr.ending_round = dune_state->GetCurrentRound();
    } else {
      gr.ending_round = -1;
    }
    gr.current_vp = dune_state
                    ? dune_state->GetPlayerVpForTesting(model_player)
                    : -1;
    gr.final_scored_vp = dune_state
                         ? dune_state->FinalScoredVp(model_player)
                         : -1;

    // PWO-5 section 13.5: every seat's TRUE final VP, and the exact margin.
    if (dune_state) {
      long long opponent_sum = 0;
      for (int p = 0; p < kNumPlayers; ++p) {
        gr.final_scored_vp_all[p] = dune_state->FinalScoredVp(p);
        gr.player_leaders[p] = dune_state->PlayerLeader(p);
        if (p != model_player) opponent_sum += gr.final_scored_vp_all[p];
      }
      // The mean of the THREE opponents, not the best of them.
      gr.vp_margin = static_cast<double>(gr.final_scored_vp_all[model_player]) -
                     static_cast<double>(opponent_sum) /
                         static_cast<double>(kNumPlayers - 1);
      gr.vp_margin_valid = true;
    } else {
      gr.final_scored_vp_all.fill(-1);
      gr.player_leaders.fill(-1);
      gr.vp_margin = 0.0;
      gr.vp_margin_valid = false;
    }

    // --- Phase-3 pace/threshold telemetry ---
    gr.has_r5 = round_end_seen[5];
    gr.has_r6 = round_end_seen[6];
    gr.has_r7 = round_end_seen[7];
    if (gr.has_r5) gr.vp_end_r5 = vp_at_round_end[5];
    if (gr.has_r6) gr.vp_end_r6 = vp_at_round_end[6];
    if (gr.has_r7) gr.vp_end_r7 = vp_at_round_end[7];
    if (gr.has_r7) {
      gr.r7_track_vp = gr.vp_end_r7;
      gr.has_r7_track_vp = true;
    } else if (dune_state != nullptr) {
      for (int p = 0; p < kNumPlayers; ++p) {
        gr.r7_track_vp[p] = dune_state->GetPlayerVp(p);
      }
      gr.has_r7_track_vp = true;
    }
    gr.end_trigger_player = -1;
    gr.end_trigger_round = -1;
    if (dune_state) {
      // Game end is checked only at round boundaries: at the terminal round-end
      // a player triggered the end iff running VP >= kTargetVp. Among crossers
      // pick the leader (max VP; tie -> lowest seat).
      int best_p = -1, best_vp = -1;
      for (int p = 0; p < kNumPlayers; ++p) {
        int v = dune_state->GetPlayerVp(p);
        if (v >= kTargetVp && v > best_vp) {
          best_vp = v;
          best_p = p;
        }
      }
      if (best_p >= 0) {
        gr.end_trigger_player = best_p;
        gr.end_trigger_round = gr.ending_round;
      }
    }
    // First place = argmax of placement returns (tie -> lowest seat).
    int first_place = 0;
    for (int p = 1; p < kNumPlayers; ++p) {
      if (returns[p] > returns[first_place]) first_place = p;
    }
    gr.winner_triggered = (gr.end_trigger_player >= 0)
                              ? (first_place == gr.end_trigger_player ? 1 : 0)
                              : -1;

    // --- WO-1 Phase 3 pace/tempo telemetry ---------------------------------
    // Dense round-end VP, replacing the r5/r6/r7 ceiling. The snapshot hook
    // already captured rounds 1..kMaxRounds; only the emission was truncated.
    gr.rounds_captured = 0;
    for (int rd = 1; rd <= kMaxRounds; ++rd) {
      if (!round_end_seen[rd]) break;
      gr.vp_end_by_round[rd - 1] = vp_at_round_end[rd];
      gr.solari_end_by_round[rd - 1] = solari_at_round_end[rd];
      gr.rounds_captured = rd;
    }
    for (int p = 0; p < kNumPlayers; ++p) {
      gr.first_round_vp_ge_11[p] = -1;
      for (int rd = 1; rd <= gr.rounds_captured; ++rd) {
        if (vp_at_round_end[rd][p] >= kTargetVp) {
          gr.first_round_vp_ge_11[p] = rd;
          break;
        }
      }
    }
    gr.reveals_with_agents_remaining = reveals_with_agents_remaining;
    gr.specimen_conversions = specimen_conversions;
    gr.atomics_used = atomics_used;
    gr.atomics_in_reveal = atomics_in_reveal;
    gr.atomics_persuasion = atomics_persuasion;
    gr.atomics_round = atomics_round;
    gr.plot_plays = plot_plays;
    gr.plot_pre_hand_sizes = plot_pre_hand_sizes;
    if (dune_state != nullptr) {
      for (int p = 0; p < kNumPlayers; ++p) {
        gr.research_col[p] = dune_state->GetResearchBottomColForTesting(p);
        gr.research_row[p] = dune_state->GetResearchBottomRowForTesting(p);
      }
    }
    gr.research_paths = game_research_paths;
    gr.terminal_threshold_set.clear();
    gr.candidate_in_terminal_threshold_set = false;
    gr.terminal_reason = "other";
    if (dune_state) {
      for (int p = 0; p < kNumPlayers; ++p) {
        if (dune_state->GetPlayerVp(p) >= kTargetVp) {
          gr.terminal_threshold_set.push_back(p);
          if (p == model_player) gr.candidate_in_terminal_threshold_set = true;
        }
      }
      // Mirrors the engine's own end test (dune_imperium.cc:8891-8904):
      // end_game <- (round_ > kMaxRounds) OR (any vp_[p] >= kTargetVp).
      // Threshold takes precedence when both hold, matching the precedence
      // end_trigger_player already uses. "other" is unreachable; if it ever
      // appears, the engine's end condition changed under a frozen engine.
      if (!gr.terminal_threshold_set.empty()) {
        gr.terminal_reason = "threshold";
      } else if (dune_state->GetCurrentRound() >= kMaxRounds) {
        gr.terminal_reason = "round_limit";
      }
    }

    if (record_market_diagnostics && market_results != nullptr) {
      (*market_results)[result_index] = market_tracker.GetData();
    }
  }
}

void RunEvaluationGames(
    const EvalConfig& cfg,
    const EvalRuntime& rt,
    std::ofstream* dump_out,
    std::vector<GameResult>& results,
    std::vector<dune_market_diag::CompactPerGameMarketDiagnostics>* market_results) {
  std::atomic<int> next_game_id{0};
  std::vector<std::thread> threads;
  for (unsigned int t = 0; t < rt.num_threads; ++t) {
    threads.emplace_back(
        WorkerThread, static_cast<int>(t), rt.game, rt.model_evaluator,
        std::cref(rt.opponent_evaluators), std::cref(rt.opponent_names), rt.obs_size,
        rt.main_input_dim, rt.cand_market_mode, std::cref(rt.opp_metadata),
        rt.provides_info_state_tensor, rt.provides_observations_tensor,
        std::ref(next_game_id), cfg.start_episode_id, cfg.total_games,
        cfg.base_seed, cfg.domain, cfg.greedy, cfg.temperature,
        cfg.has_opponent_greedy, cfg.opponent_greedy,
        cfg.has_opponent_temperature, cfg.opponent_temperature,
        static_cast<float>(cfg.candidate_logit_cap),
        static_cast<float>(cfg.opponent_logit_cap),
        dump_out,
        std::ref(results),
        cfg.record_market_diagnostics,
        market_results,
        cfg.record_history);
  }
  for (auto& th : threads) {
    if (th.joinable()) th.join();
  }
}

}  // namespace dune_pop_eval
}  // namespace open_spiel
