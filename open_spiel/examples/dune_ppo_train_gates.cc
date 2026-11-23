#include "dune_ppo_train_internal.h"

namespace open_spiel::ppo_train_internal {

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

std::vector<R755Gate2Diagnostics> g_gate2_diagnostics;

void WriteR755Gate2Json(const std::string& path, const std::vector<R755Gate2Diagnostics>& diag) {
  if (path.empty()) return;
  json::Array updates;
  for (const auto& d : diag) {
    json::Object u;
    u["update"] = static_cast<int64_t>(d.update);
    u["total_transitions"] = d.total_transitions;
    u["single_action_transitions"] = d.single_action_transitions;
    u["multi_action_transitions"] = d.multi_action_transitions;
    u["override_masked_transitions"] = d.override_masked_transitions;
    u["ppo_multi_action_transitions"] = d.ppo_multi_action_transitions;
    u["mask_identity_holds"] = d.mask_identity_holds;
    u["nan_inf_count"] = d.nan_inf_count;
    u["logprob_recompute_max_abs_diff"] = d.logprob_recompute_max_abs_diff;
    u["collector_vs_learner_mean_diff"] = d.collector_vs_learner_mean_diff;
    u["collector_vs_learner_p99_diff"] = d.collector_vs_learner_p99_diff;
    u["collector_vs_learner_max_diff"] = d.collector_vs_learner_max_diff;
    u["collector_vs_learner_kl"] = d.collector_vs_learner_kl;
    u["mean_student_kl"] = d.mean_student_kl;
    u["mean_target_kl"] = d.mean_target_kl;
    u["mean_delta_p_target"] = d.mean_delta_p_target;
    u["mean_delta_p_student"] = d.mean_delta_p_student;
    u["student_to_target_ratio"] = d.student_to_target_ratio;
    u["mean_non_override_kl"] = d.mean_non_override_kl;
    u["pre_step_logprob_mean_abs_diff"] = d.pre_step_logprob_mean_abs_diff;
    u["pre_step_logprob_p99_abs_diff"] = d.pre_step_logprob_p99_abs_diff;
    u["pre_step_logprob_max_abs_diff"] = d.pre_step_logprob_max_abs_diff;
    u["pre_step_ratio_min"] = d.pre_step_ratio_min;
    u["pre_step_ratio_p01"] = d.pre_step_ratio_p01;
    u["pre_step_ratio_p05"] = d.pre_step_ratio_p05;
    u["pre_step_ratio_p50"] = d.pre_step_ratio_p50;
    u["pre_step_ratio_p95"] = d.pre_step_ratio_p95;
    u["pre_step_ratio_p99"] = d.pre_step_ratio_p99;
    u["pre_step_ratio_max"] = d.pre_step_ratio_max;
    u["pre_step_clip_frac"] = d.pre_step_clip_frac;
    u["aux_distinct_visited"] = d.aux_distinct_visited;
    u["aux_total_presentations"] = d.aux_total_presentations;
    u["visited_aux_count"] = d.visited_aux_count;
    u["unvisited_aux_count"] = d.unvisited_aux_count;
    u["sum_delta_p_target"] = d.sum_delta_p_target;
    u["sum_delta_p_student"] = d.sum_delta_p_student;
    u["visited_sum_delta_p_target"] = d.visited_sum_delta_p_target;
    u["visited_sum_delta_p_student"] = d.visited_sum_delta_p_student;
    u["unvisited_sum_delta_p_target"] = d.unvisited_sum_delta_p_target;
    u["unvisited_sum_delta_p_student"] = d.unvisited_sum_delta_p_student;
    u["visited_student_to_target_ratio"] = d.visited_student_to_target_ratio;
    u["unvisited_student_to_target_ratio"] = d.unvisited_student_to_target_ratio;
    for (int c = 1; c <= 5; ++c) {
      u[absl::StrCat("cat", c, "_searched")] = d.searched_by_cat[c];
      u[absl::StrCat("cat", c, "_executed_overrides")] = d.executed_overrides_by_cat[c];
    }
    u["cat3_aux_distinct_visited_fraction"] = d.cat3_aux_distinct_visited_fraction;
    u["imperium_buy_rate"] = d.imperium_buy_rate;
    u["tleilaxu_buy_rate"] = d.tleilaxu_buy_rate;
    u["tech_buy_rate"] = d.tech_buy_rate;
    u["mean_unspent_persuasion"] = d.mean_unspent_persuasion;
    u["purchase_grad_cosine"] = d.purchase_grad_cosine;
    json::Array arr_i;
    for (double val : d.cat3_i_avg) arr_i.emplace_back(val);
    u["cat3_i_avg"] = std::move(arr_i);
    json::Array arr_ii;
    for (double val : d.cat3_ii_avg) arr_ii.emplace_back(val);
    u["cat3_ii_avg"] = std::move(arr_ii);
    json::Array arr_iii;
    for (double val : d.cat3_iii_avg) arr_iii.emplace_back(val);
    u["cat3_iii_avg"] = std::move(arr_iii);
    json::Array arr_counts;
    for (int64_t val : d.cat3_bucket_counts) arr_counts.emplace_back(val);
    u["cat3_bucket_counts"] = std::move(arr_counts);
    updates.push_back(std::move(u));
  }
  json::Object root;
  root["gate"] = "Gate 2";
  root["updates"] = std::move(updates);
  std::filesystem::path p(path);
  if (!p.parent_path().empty()) std::filesystem::create_directories(p.parent_path());
  const std::string tmp_path = path + ".tmp";
  std::ofstream ofs(tmp_path);
  if (ofs.is_open()) {
    ofs << json::ToString(root, true) << "\n";
    ofs.flush();
    ofs.close();
    std::filesystem::rename(tmp_path, path);
    std::cout << "[R755-GATE2] Wrote diagnostics to " << path << "\n";
  }
}

void RunR755Gate1Audit(
    const Game* game,
    std::shared_ptr<IGameEvaluator> evaluator,
    int64_t obs_size,
    uint64_t master_seed,
    int rollout_games,
    const std::string& output_path) {
  auto batched_eval = std::dynamic_pointer_cast<open_spiel::BatchedEvaluator>(evaluator);
  std::shared_ptr<open_spiel::BatchedNNEvaluator> nn_eval = nullptr;
  if (batched_eval != nullptr) {
    nn_eval = std::make_shared<open_spiel::BatchedNNEvaluator>(
        batched_eval, static_cast<float>(absl::GetFlag(FLAGS_logit_cap)),
        batched_eval->MarketMode(),
        /*forward_active_rollouts=*/false);
  }
  std::vector<std::shared_ptr<open_spiel::BatchedNNEvaluator>> search_evals;
  if (nn_eval) search_evals.push_back(nn_eval);

  int num_games = rollout_games > 0 ? rollout_games : 40;
  std::cout << "[R755-GATE1] Running " << num_games << "-game audit on CPU/accelerator...\n";

  int64_t N_searched = 0;
  int64_t N_accepted = 0;
  std::vector<int64_t> n_searched_by_cat(6, 0);
  std::vector<int64_t> n_accepted_by_cat(6, 0);
  std::vector<int64_t> executed_overrides_all_seats(6, 0);
  std::vector<int64_t> game_N_searched(num_games, 0);
  std::vector<int64_t> game_N_accepted(num_games, 0);
  std::vector<int64_t> sum_rollout_steps_by_cat(6, 0);
  std::vector<int64_t> sum_rollout_evals_by_cat(6, 0);
  double sum_delta_p_accepted = 0.0;
  std::vector<Gate1AuditedRoot> audited_roots;

  auto eval_market_mode = evaluator ? evaluator->MarketMode() : dune_imperium::MarketAppendixMode::kNone;
  auto eval_deck_semantics = evaluator ? evaluator->DeckPoolSemantics() : dune_imperium::DeckPoolSemantics::kLegacy;

  struct PerGameStats {
    int64_t N_searched = 0;
    int64_t N_accepted = 0;
    std::vector<int64_t> n_searched_by_cat = std::vector<int64_t>(6, 0);
    std::vector<int64_t> n_accepted_by_cat = std::vector<int64_t>(6, 0);
    std::vector<int64_t> executed_overrides_all_seats = std::vector<int64_t>(6, 0);
    std::vector<int64_t> sum_rollout_steps_by_cat = std::vector<int64_t>(6, 0);
    std::vector<int64_t> sum_rollout_evals_by_cat = std::vector<int64_t>(6, 0);
    double sum_delta_p_accepted = 0.0;
    std::vector<Gate1AuditedRoot> local_roots;
  };
  std::vector<PerGameStats> per_game(num_games);

  std::vector<std::future<void>> game_futures;
  game_futures.reserve(num_games);
  for (int episode_id = 0; episode_id < num_games; ++episode_id) {
    game_futures.push_back(
        open_spiel::dune_imperium::SearchRolloutThreadPool::Instance().Submit([&, episode_id]() {
          auto& g = per_game[episode_id];
          int audited_seat = episode_id % 4;
          auto state = game->NewInitialState();
          auto chance_rng = dune_seed::MakeRng64(dune_seed::DeriveSeed(master_seed, dune_seed::kDomainTrain, episode_id, dune_seed::kStreamChance));
          std::mt19937_64 policy_rng[4];
          for (int p = 0; p < 4; ++p) {
            policy_rng[p] = dune_seed::MakeRng64(dune_seed::DeriveSeed(master_seed, dune_seed::kDomainTrain, episode_id, dune_seed::kStreamPolicyPlayer0 + p));
          }

          open_spiel::dune_imperium::TurnSearchTree turn_trees[4];
          bool space_placed_this_turn[4] = {false, false, false, false};
          bool turn_coin_active[4] = {false, false, false, false};
          Player last_turn_player = open_spiel::kInvalidPlayer;
          Action last_action = open_spiel::kInvalidAction;
          int64_t decision_index = 0;

          while (!state->IsTerminal()) {
            if (state->IsChanceNode()) {
              auto outcomes = state->ChanceOutcomes();
              if (outcomes.empty()) break;
              double u = std::generate_canonical<double, 53>(chance_rng);
              Action choice = open_spiel::SampleAction(outcomes, u).first;
              state->ApplyAction(choice);
              continue;
            }

            ++decision_index;
            Player current_player = state->CurrentPlayer();
            auto actions = state->LegalActions();
            if (actions.empty()) break;

            if (current_player != last_turn_player || last_action == dune_imperium::kActionEndTurn) {
              if (current_player >= 0 && current_player < 4) {
                space_placed_this_turn[current_player] = false;
                turn_trees[current_player].Reset();
                uint64_t coin_seed = dune_seed::DeriveSeed(master_seed, dune_seed::kDomainTrain, episode_id, decision_index, 0x52373535ULL);
                std::mt19937_64 coin_rng(coin_seed);
                turn_coin_active[current_player] = (std::uniform_real_distribution<double>(0.0, 1.0)(coin_rng) < absl::GetFlag(FLAGS_r755_search_prob));
              }
            }

            const auto* dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
            std::vector<float> obs(obs_size, 0.0f);
            if (dune_state != nullptr) {
              dune_state->InformationStateTensorWithAppendix(current_player, eval_market_mode, absl::MakeSpan(obs), eval_deck_semantics);
            } else if (game->GetType().provides_information_state_tensor) {
              state->InformationStateTensor(current_player, absl::MakeSpan(obs));
            } else {
              state->ObservationTensor(current_player, absl::MakeSpan(obs));
            }

            dune_semantic::CandidateActionData cand_data;
            const dune_semantic::CandidateActionData* p_cand_data = nullptr;
            if (dune_state != nullptr && evaluator != nullptr) {
              dune_semantic::ExtractCandidateDescriptors(*dune_state, actions, &cand_data, evaluator->SemanticDescriptorSchema());
              p_cand_data = &cand_data;
            }
            EvalResult result = evaluator->EvaluateWithActions(obs, p_cand_data);
            CenterAndCapLegalLogits(result.logits, actions, static_cast<float>(absl::GetFlag(FLAGS_logit_cap)));
            std::vector<double> legal_probs;
            auto sample = SampleTrainingPolicyDistribution(*state, &policy_rng[current_player], result.logits, actions, &legal_probs);
            Action action = sample.action;

            std::vector<std::pair<Action, double>> raw_policy_pairs;
            raw_policy_pairs.reserve(actions.size());
            for (size_t i = 0; i < actions.size(); ++i) {
              raw_policy_pairs.push_back({actions[i], i < legal_probs.size() ? legal_probs[i] : 0.0});
            }

            auto cat = open_spiel::dune_imperium::ClassifyR755SearchCategory(
                *state, current_player, space_placed_this_turn[current_player],
                absl::GetFlag(FLAGS_r755_category_mask));
            int cat_idx = static_cast<int>(cat);

            bool coin_active = false;
            if (cat == open_spiel::dune_imperium::R755SearchCategory::kAgentCardOrStartPlot ||
                cat == open_spiel::dune_imperium::R755SearchCategory::kImmediateSpacePlacement) {
              coin_active = turn_coin_active[current_player];
            } else if (cat != open_spiel::dune_imperium::R755SearchCategory::kIneligible) {
              uint64_t coin_seed = dune_seed::DeriveSeed(master_seed, dune_seed::kDomainTrain, episode_id, decision_index, 0x52373535ULL);
              std::mt19937_64 coin_rng(coin_seed);
              coin_active = (std::uniform_real_distribution<double>(0.0, 1.0)(coin_rng) < absl::GetFlag(FLAGS_r755_search_prob));
            }

            if (current_player == audited_seat && cat != open_spiel::dune_imperium::R755SearchCategory::kIneligible) {
              ++g.N_searched;
              ++g.n_searched_by_cat[cat_idx];

              open_spiel::dune_imperium::SearchRolloutCrnMode crn_mode =
                  open_spiel::dune_imperium::SearchRolloutCrnMode::kSequential;
              const std::string crn_str = absl::GetFlag(FLAGS_r755_crn_mode);
              if (crn_str == "sequential") {
                crn_mode = open_spiel::dune_imperium::SearchRolloutCrnMode::kSequential;
              } else if (crn_str == "deck_order") {
                crn_mode = open_spiel::dune_imperium::SearchRolloutCrnMode::kDeckOrderCrn;
              } else if (crn_str == "event_keyed") {
                crn_mode = open_spiel::dune_imperium::SearchRolloutCrnMode::kEventKeyedCrn;
              } else {
                open_spiel::SpielFatalError(absl::StrCat("Unknown r755_crn_mode: ", crn_str));
              }

              open_spiel::dune_imperium::CompoundSearchOptions opts;
              opts.precomputed_raw_prior = absl::GetFlag(FLAGS_r755_reuse_root_prior) ? &raw_policy_pairs : nullptr;
              opts.dedup_paired_worlds = absl::GetFlag(FLAGS_r755_dedup_paired_worlds);
              opts.win_speed_tiebreak_eps = open_spiel::ResolveOnlineTeacherWinSpeedEps(
                  absl::GetFlag(FLAGS_r765_teacher_win_speed_tiebreak_eps),
                  absl::GetFlag(FLAGS_win_speed_bonus_per_round));
              opts.ceiling_aware_margin =
                  absl::GetFlag(FLAGS_r765_teacher_ceiling_aware_margin);
              opts.stage1_rollouts = absl::GetFlag(FLAGS_r755_stage1_rollouts);
              opts.stage1_min_mean_diff = absl::GetFlag(FLAGS_r755_stage1_min_mean_diff);
              opts.stage1_subwave_rollouts = absl::GetFlag(FLAGS_r755_stage1_subwave_rollouts);
              opts.stage1_subwave_min_diff = absl::GetFlag(FLAGS_r755_stage1_subwave_min_diff);
              opts.center_4seat_leaf_values = absl::GetFlag(FLAGS_r755_leaf_value_centering);

              uint64_t search_seed = dune_seed::DeriveSeed(master_seed, dune_seed::kDomainTrain, episode_id, decision_index, 0x52373553ULL);
              open_spiel::dune_imperium::TurnSearchTree temp_tree;
              open_spiel::dune_imperium::CompoundSearchDecisionResult search_res =
                  open_spiel::dune_imperium::CompoundRolloutSearchDecision(
                      *state, current_player, &temp_tree, search_evals,
                      /*top_k=*/3, /*rollouts_per_action=*/16, /*min_override_margin=*/0.15,
                      open_spiel::dune_imperium::TreeReuseMode::kTargetFill, search_seed,
                      open_spiel::dune_imperium::SearchPairingMode::kPaired,
                      open_spiel::dune_imperium::SearchTruncationMode::kNextRoundSearchPlayer,
                      crn_mode, /*world_sampler=*/nullptr,
                      absl::GetFlag(FLAGS_r755_min_t_stat), opts);
              if (coin_active && search_res.overridden && absl::GetFlag(FLAGS_r755_execute_overrides)) {
                action = search_res.chosen_action;
                ++g.executed_overrides_all_seats[cat_idx];
              }

              if (search_res.overridden) {
                ++g.N_accepted;
                ++g.n_accepted_by_cat[cat_idx];
                g.sum_rollout_steps_by_cat[cat_idx] += search_res.rollout_steps;
                g.sum_rollout_evals_by_cat[cat_idx] += search_res.rollout_nn_evals;

                int round_num = dune_state ? dune_state->GetCurrentRound() : 1;
                int bucket = round_num <= 3 ? 0 : (round_num <= 6 ? 1 : 2);
                auto mpo_target = open_spiel::dune_imperium::ComputePairwiseMpoTarget(
                    raw_policy_pairs, actions, search_res.raw_action, search_res.raw_mean_utility,
                    search_res.best_action, search_res.chosen_mean_utility, 0.15, 0.05);
                double delta_p = 0.0;
                for (size_t a_i = 0; a_i < actions.size(); ++a_i) {
                  if (actions[a_i] == search_res.best_action) {
                    delta_p = mpo_target[a_i] - (a_i < legal_probs.size() ? legal_probs[a_i] : 0.0);
                    break;
                  }
                }
                g.sum_delta_p_accepted += delta_p;

                Gate1AuditedRoot r;
                r.game_id = episode_id;
                r.category = cat_idx;
                r.round = round_num;
                r.round_bucket = bucket;
                r.state_clone = state->Clone();
                r.raw_action = search_res.raw_action;
                r.best_action = search_res.best_action;
                r.delta_p = delta_p;
                r.raw_policy = raw_policy_pairs;
                r.rollout_steps = search_res.rollout_steps;
                r.rollout_nn_evals = search_res.rollout_nn_evals;
                g.local_roots.push_back(std::move(r));
              }
            } else if (current_player != audited_seat && cat != open_spiel::dune_imperium::R755SearchCategory::kIneligible && coin_active) {
              open_spiel::dune_imperium::SearchRolloutCrnMode crn_mode =
                  open_spiel::dune_imperium::SearchRolloutCrnMode::kSequential;
              const std::string crn_str = absl::GetFlag(FLAGS_r755_crn_mode);
              if (crn_str == "sequential") {
                crn_mode = open_spiel::dune_imperium::SearchRolloutCrnMode::kSequential;
              } else if (crn_str == "deck_order") {
                crn_mode = open_spiel::dune_imperium::SearchRolloutCrnMode::kDeckOrderCrn;
              } else if (crn_str == "event_keyed") {
                crn_mode = open_spiel::dune_imperium::SearchRolloutCrnMode::kEventKeyedCrn;
              } else {
                open_spiel::SpielFatalError(absl::StrCat("Unknown r755_crn_mode: ", crn_str));
              }

              open_spiel::dune_imperium::CompoundSearchOptions opts;
              opts.precomputed_raw_prior = absl::GetFlag(FLAGS_r755_reuse_root_prior) ? &raw_policy_pairs : nullptr;
              opts.dedup_paired_worlds = absl::GetFlag(FLAGS_r755_dedup_paired_worlds);
              opts.win_speed_tiebreak_eps = open_spiel::ResolveOnlineTeacherWinSpeedEps(
                  absl::GetFlag(FLAGS_r765_teacher_win_speed_tiebreak_eps),
                  absl::GetFlag(FLAGS_win_speed_bonus_per_round));
              opts.ceiling_aware_margin =
                  absl::GetFlag(FLAGS_r765_teacher_ceiling_aware_margin);
              opts.stage1_rollouts = absl::GetFlag(FLAGS_r755_stage1_rollouts);
              opts.stage1_min_mean_diff = absl::GetFlag(FLAGS_r755_stage1_min_mean_diff);
              opts.stage1_subwave_rollouts = absl::GetFlag(FLAGS_r755_stage1_subwave_rollouts);
              opts.stage1_subwave_min_diff = absl::GetFlag(FLAGS_r755_stage1_subwave_min_diff);
              opts.center_4seat_leaf_values = absl::GetFlag(FLAGS_r755_leaf_value_centering);

              uint64_t search_seed = dune_seed::DeriveSeed(master_seed, dune_seed::kDomainTrain, episode_id, decision_index, 0x52373553ULL);
              open_spiel::dune_imperium::TurnSearchTree temp_tree;
              auto search_res = open_spiel::dune_imperium::CompoundRolloutSearchDecision(
                  *state, current_player, &temp_tree, search_evals,
                  /*top_k=*/3, /*rollouts_per_action=*/16, /*min_override_margin=*/0.15,
                  open_spiel::dune_imperium::TreeReuseMode::kTargetFill, search_seed,
                  open_spiel::dune_imperium::SearchPairingMode::kPaired,
                  open_spiel::dune_imperium::SearchTruncationMode::kNextRoundSearchPlayer,
                  crn_mode, /*world_sampler=*/nullptr,
                  absl::GetFlag(FLAGS_r755_min_t_stat), opts);
              if (search_res.overridden && absl::GetFlag(FLAGS_r755_execute_overrides)) {
                action = search_res.chosen_action;
                ++g.executed_overrides_all_seats[cat_idx];
              }
            }

            if (current_player >= 0 && current_player < 4) {
              if (action == dune_imperium::kActionEndTurn) {
                turn_trees[current_player].Reset();
              } else {
                turn_trees[current_player].Advance(action);
              }
              if (action >= dune_imperium::kActionAgentSpaceConspire &&
                  action <= dune_imperium::kActionAgentSpaceSietchTabr) {
                space_placed_this_turn[current_player] = true;
              }
              last_turn_player = current_player;
              last_action = action;
            }
            state->ApplyAction(action);
          }
        }));
  }

  for (auto& fut : game_futures) {
    fut.get();
  }

  for (int episode_id = 0; episode_id < num_games; ++episode_id) {
    auto& g = per_game[episode_id];
    N_searched += g.N_searched;
    N_accepted += g.N_accepted;
    game_N_searched[episode_id] = g.N_searched;
    game_N_accepted[episode_id] = g.N_accepted;
    for (int c = 0; c < 6; ++c) {
      n_searched_by_cat[c] += g.n_searched_by_cat[c];
      n_accepted_by_cat[c] += g.n_accepted_by_cat[c];
      executed_overrides_all_seats[c] += g.executed_overrides_all_seats[c];
      sum_rollout_steps_by_cat[c] += g.sum_rollout_steps_by_cat[c];
      sum_rollout_evals_by_cat[c] += g.sum_rollout_evals_by_cat[c];
    }
    sum_delta_p_accepted += g.sum_delta_p_accepted;
    for (auto& r : g.local_roots) {
      audited_roots.push_back(std::move(r));
    }
  }

  std::cout << "[R755-GATE1] Collection complete: N_searched=" << N_searched
            << ", N_accepted=" << N_accepted << "\n";

  // Stratified Ref512 selection (cap 120)
  std::vector<size_t> sampled_indices;
  if (audited_roots.size() <= 120) {
    for (size_t i = 0; i < audited_roots.size(); ++i) {
      audited_roots[i].inclusion_prob = 1.0;
      sampled_indices.push_back(i);
    }
  } else {
    std::map<std::pair<int, int>, std::vector<size_t>> strata;
    for (size_t i = 0; i < audited_roots.size(); ++i) {
      strata[{audited_roots[i].category, audited_roots[i].round_bucket}].push_back(i);
    }
    const int num_strata = static_cast<int>(strata.size());
    const int per_stratum_reserve = (num_strata > 0) ? std::min(15, 120 / num_strata) : 15;
    std::map<std::pair<int, int>, int> n_sampled;
    int reserved_total = 0;
    int excess_total = 0;
    for (const auto& [k, v] : strata) {
      if (static_cast<int>(v.size()) <= per_stratum_reserve) {
        n_sampled[k] = static_cast<int>(v.size());
        reserved_total += static_cast<int>(v.size());
      } else {
        n_sampled[k] = per_stratum_reserve;
        reserved_total += per_stratum_reserve;
        excess_total += (static_cast<int>(v.size()) - per_stratum_reserve);
      }
    }
    int remaining_cap = 120 - reserved_total;
    if (remaining_cap > 0 && excess_total > 0) {
      int allocated = 0;
      std::vector<std::pair<double, std::pair<int, int>>> remainders;
      for (const auto& [k, v] : strata) {
        if (static_cast<int>(v.size()) > per_stratum_reserve) {
          double exact = static_cast<double>(remaining_cap) * (static_cast<int>(v.size()) - per_stratum_reserve) / static_cast<double>(excess_total);
          int add = static_cast<int>(std::floor(exact));
          n_sampled[k] += add;
          allocated += add;
          remainders.push_back({exact - add, k});
        }
      }
      std::sort(remainders.rbegin(), remainders.rend());
      for (int r = 0; r < remaining_cap - allocated && r < static_cast<int>(remainders.size()); ++r) {
        if (n_sampled[remainders[r].second] < static_cast<int>(strata.at(remainders[r].second).size())) {
          n_sampled[remainders[r].second]++;
        }
      }
    }
    for (auto& [k, v] : strata) {
      n_sampled[k] = std::min(n_sampled[k], static_cast<int>(v.size()));
      int n_h = n_sampled[k];
      int N_h = static_cast<int>(v.size());
      std::mt19937_64 stratum_rng(20261102ULL + k.first * 100 + k.second);
      std::shuffle(v.begin(), v.end(), stratum_rng);
      double prob = static_cast<double>(n_h) / static_cast<double>(N_h);
      for (int j = 0; j < n_h; ++j) {
        audited_roots[v[j]].inclusion_prob = prob;
        sampled_indices.push_back(v[j]);
      }
    }
  }

  std::cout << "[R755-GATE1] Running Ref512 rollouts on " << sampled_indices.size() << " sampled roots...\n";

  // Ref512 paired terminal rollouts
  auto batcher = batched_eval;
  const size_t num_sampled = sampled_indices.size();
  std::vector<std::vector<double>> ret_best_mat(num_sampled, std::vector<double>(512, 0.0));
  std::vector<std::vector<double>> ret_greedy_mat(num_sampled, std::vector<double>(512, 0.0));
  std::vector<std::vector<double>> ret_pi_mat(num_sampled, std::vector<double>(512, 0.0));

  const size_t total_ref_tasks = num_sampled * 512;
  const int num_ref_workers = static_cast<int>(std::min<size_t>(512, total_ref_tasks));
  if (batcher != nullptr && num_ref_workers > 0) {
    batcher->AdjustActiveRollouts(num_ref_workers);
  }
  std::atomic<size_t> next_ref_task{0};
  std::vector<std::future<void>> ref_futures;
  ref_futures.reserve(num_ref_workers);
  for (int worker_id = 0; worker_id < num_ref_workers; ++worker_id) {
    ref_futures.push_back(open_spiel::dune_imperium::SearchRolloutThreadPool::Instance().Submit([&]() {
      open_spiel::dune_imperium::ActiveRolloutReleaseGuard release_guard(batcher.get(), 1);
      for (;;) {
        const size_t task_idx = next_ref_task.fetch_add(1, std::memory_order_relaxed);
        if (task_idx >= total_ref_tasks) break;
        const size_t r_k = task_idx / 512;
        const int w = static_cast<int>(task_idx % 512);
        const size_t s_idx = sampled_indices[r_k];
        const auto& root = audited_roots[s_idx];
        const auto* dune = dynamic_cast<const dune_imperium::DuneImperiumState*>(root.state_clone.get());
        Player p = root.state_clone->CurrentPlayer();

        uint64_t wseed = dune_seed::DeriveSeed(20261102ULL, s_idx, w);
        std::mt19937_64 rng(wseed);
        auto rng_func = [&rng]() { return std::generate_canonical<double, 53>(rng); };
        std::unique_ptr<State> world;
        if (dune) world = dune->ResampleFromInfostate(p, rng_func);
        if (!world) world = root.state_clone->Clone();

        // Branch 1: a_greedy
        auto sim1 = world->Clone();
        sim1->ApplyAction(root.raw_action);
        auto traj1 = open_spiel::dune_imperium::SimulateRemainingTurn(
            sim1.get(), p, nn_eval.get(), wseed + 999, open_spiel::dune_imperium::SearchTruncationMode::kTerminal);
        double ret_greedy = traj1.terminal_utility;

        // Branch 2: a*
        auto sim2 = world->Clone();
        sim2->ApplyAction(root.best_action);
        auto traj2 = open_spiel::dune_imperium::SimulateRemainingTurn(
            sim2.get(), p, nn_eval.get(), wseed + 999, open_spiel::dune_imperium::SearchTruncationMode::kTerminal);
        double ret_best = traj2.terminal_utility;

        // Branch 3: pi_old
        uint64_t pi_seed = dune_seed::DeriveSeed(20261102ULL, s_idx, w, 0x5049ULL);
        std::mt19937_64 pi_rng(pi_seed);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double u = dist(pi_rng);
        Action a_w = root.raw_policy.front().first;
        double cum = 0.0;
        for (const auto& ap : root.raw_policy) {
          cum += ap.second;
          if (u <= cum) {
            a_w = ap.first;
            break;
          }
        }
        double ret_pi = 0.0;
        if (a_w == root.raw_action) {
          ret_pi = ret_greedy;
        } else if (a_w == root.best_action) {
          ret_pi = ret_best;
        } else {
          auto sim3 = world->Clone();
          sim3->ApplyAction(a_w);
          auto traj3 = open_spiel::dune_imperium::SimulateRemainingTurn(
              sim3.get(), p, nn_eval.get(), wseed + 999, open_spiel::dune_imperium::SearchTruncationMode::kTerminal);
          ret_pi = traj3.terminal_utility;
        }

        ret_best_mat[r_k][w] = ret_best;
        ret_greedy_mat[r_k][w] = ret_greedy;
        ret_pi_mat[r_k][w] = ret_pi;
      }
    }));
  }
  for (auto& f : ref_futures) {
    f.get();
  }
  for (size_t r_k = 0; r_k < num_sampled; ++r_k) {
    auto& root = audited_roots[sampled_indices[r_k]];
    double sum_ret_best = 0.0, sum_ret_greedy = 0.0, sum_ret_pi = 0.0;
    for (int w = 0; w < 512; ++w) {
      sum_ret_best += ret_best_mat[r_k][w];
      sum_ret_greedy += ret_greedy_mat[r_k][w];
      sum_ret_pi += ret_pi_mat[r_k][w];
    }
    root.delta_q = (sum_ret_best - sum_ret_greedy) / 512.0;
    root.delta_v_exec = (sum_ret_best - sum_ret_pi) / 512.0;
  }

  // Horvitz-Thompson Estimators
  double sum_ht_action = 0.0, sum_ht_exec = 0.0, sum_ht_target = 0.0, sum_ht_neg = 0.0;
  for (size_t s_idx : sampled_indices) {
    const auto& root = audited_roots[s_idx];
    double r_i = root.inclusion_prob;
    sum_ht_action += (root.delta_q / r_i);
    sum_ht_exec += (root.delta_v_exec / r_i);
    sum_ht_target += ((root.delta_p * root.delta_q) / r_i);
    if (root.delta_q < 0.0) sum_ht_neg += (1.0 / r_i);
  }
  double G_action = N_accepted > 0 ? (sum_ht_action / static_cast<double>(N_accepted)) : 0.0;
  double G_exec = N_accepted > 0 ? (sum_ht_exec / static_cast<double>(N_accepted)) : 0.0;
  double G_target = N_searched > 0 ? (sum_ht_target / static_cast<double>(N_searched)) : 0.0;
  double neg_override_rate = N_accepted > 0 ? (sum_ht_neg / static_cast<double>(N_accepted)) : 0.0;
  double mean_delta_p = N_accepted > 0 ? (sum_delta_p_accepted / static_cast<double>(N_accepted)) : 0.0;

  // 10,000 Game-Clustered Bootstrap Resamples
  int B_resamples = 10000;
  std::vector<double> boot_G_action(B_resamples, 0.0);
  std::vector<double> boot_G_exec(B_resamples, 0.0);
  std::vector<double> boot_G_target(B_resamples, 0.0);
  std::vector<double> boot_neg_rate(B_resamples, 0.0);

  for (int b = 0; b < B_resamples; ++b) {
    uint64_t bseed = dune_seed::DeriveSeed(20261102ULL, 0x424F4F54ULL, b);
    std::mt19937_64 brng(bseed);
    std::uniform_int_distribution<int> gdist(0, num_games - 1);
    std::vector<int> game_weights(num_games, 0);
    for (int g = 0; g < num_games; ++g) game_weights[gdist(brng)]++;

    double b_act = 0.0, b_exec = 0.0, b_tgt = 0.0, b_neg = 0.0;
    double b_N_acc = 0.0, b_N_src = 0.0;
    for (int g = 0; g < num_games; ++g) {
      b_N_acc += game_weights[g] * game_N_accepted[g];
      b_N_src += game_weights[g] * game_N_searched[g];
    }
    for (size_t s_idx : sampled_indices) {
      const auto& root = audited_roots[s_idx];
      int w = game_weights[root.game_id];
      if (w == 0) continue;
      double r_i = root.inclusion_prob;
      b_act += w * (root.delta_q / r_i);
      b_exec += w * (root.delta_v_exec / r_i);
      b_tgt += w * ((root.delta_p * root.delta_q) / r_i);
      if (root.delta_q < 0.0) b_neg += w * (1.0 / r_i);
    }
    boot_G_action[b] = b_N_acc > 0 ? (b_act / b_N_acc) : 0.0;
    boot_G_exec[b] = b_N_acc > 0 ? (b_exec / b_N_acc) : 0.0;
    boot_G_target[b] = b_N_src > 0 ? (b_tgt / b_N_src) : 0.0;
    boot_neg_rate[b] = b_N_acc > 0 ? (b_neg / b_N_acc) : 0.0;
  }
  std::sort(boot_G_action.begin(), boot_G_action.end());
  std::sort(boot_G_exec.begin(), boot_G_exec.end());
  std::sort(boot_G_target.begin(), boot_G_target.end());
  std::sort(boot_neg_rate.begin(), boot_neg_rate.end());

  json::Object root_obj;
  root_obj["gate"] = "Gate 1";
  root_obj["num_games"] = static_cast<int64_t>(num_games);
  root_obj["N_searched"] = N_searched;
  root_obj["N_accepted"] = N_accepted;
  root_obj["N_sampled"] = static_cast<int64_t>(sampled_indices.size());
  root_obj["mean_delta_p"] = mean_delta_p;
  root_obj["G_action"] = G_action;
  root_obj["G_action_ci95_low"] = boot_G_action[250];
  root_obj["G_action_ci95_high"] = boot_G_action[9750];
  root_obj["G_action_lcb95"] = boot_G_action[500];
  root_obj["G_exec"] = G_exec;
  root_obj["G_exec_ci95_low"] = boot_G_exec[250];
  root_obj["G_exec_ci95_high"] = boot_G_exec[9750];
  root_obj["G_exec_lcb95"] = boot_G_exec[500];
  root_obj["G_target"] = G_target;
  root_obj["G_target_ci95_low"] = boot_G_target[250];
  root_obj["G_target_ci95_high"] = boot_G_target[9750];
  root_obj["G_target_lcb95"] = boot_G_target[500];
  root_obj["neg_override_rate"] = neg_override_rate;
  root_obj["neg_override_rate_ci95_low"] = boot_neg_rate[250];
  root_obj["neg_override_rate_ci95_high"] = boot_neg_rate[9750];

  json::Array categories_json;
  for (int cat = 1; cat <= 5; ++cat) {
    int64_t c_searched = n_searched_by_cat[cat];
    int64_t c_accepted = n_accepted_by_cat[cat];
    int64_t c_sampled = 0;
    int64_t c_executed_all = executed_overrides_all_seats[cat];
    double c_ht_act = 0.0, c_ht_exec = 0.0, c_ht_tgt = 0.0, c_ht_neg = 0.0;
    for (size_t s_idx : sampled_indices) {
      const auto& r = audited_roots[s_idx];
      if (r.category == cat) {
        ++c_sampled;
        double r_i = r.inclusion_prob;
        c_ht_act += (r.delta_q / r_i);
        c_ht_exec += (r.delta_v_exec / r_i);
        c_ht_tgt += ((r.delta_p * r.delta_q) / r_i);
        if (r.delta_q < 0.0) c_ht_neg += (1.0 / r_i);
      }
    }
    double c_G_act = c_accepted > 0 ? (c_ht_act / static_cast<double>(c_accepted)) : 0.0;
    double c_G_exec = c_accepted > 0 ? (c_ht_exec / static_cast<double>(c_accepted)) : 0.0;
    double c_G_tgt = c_searched > 0 ? (c_ht_tgt / static_cast<double>(c_searched)) : 0.0;
    double c_neg_rate = c_accepted > 0 ? (c_ht_neg / static_cast<double>(c_accepted)) : 0.0;
    double c_override_rate = c_searched > 0 ? (static_cast<double>(c_accepted) / static_cast<double>(c_searched)) : 0.0;
    double c_mean_steps = c_accepted > 0 ? (static_cast<double>(sum_rollout_steps_by_cat[cat]) / static_cast<double>(c_accepted)) : 0.0;
    double c_mean_evals = c_accepted > 0 ? (static_cast<double>(sum_rollout_evals_by_cat[cat]) / static_cast<double>(c_accepted)) : 0.0;

    json::Object c_obj;
    c_obj["category_id"] = static_cast<int64_t>(cat);
    c_obj["category_name"] = open_spiel::dune_imperium::R755SearchCategoryName(static_cast<open_spiel::dune_imperium::R755SearchCategory>(cat));
    c_obj["N_searched"] = c_searched;
    c_obj["N_accepted"] = c_accepted;
    c_obj["N_sampled"] = c_sampled;
    c_obj["executed_overrides_all_seats"] = c_executed_all;
    c_obj["override_rate"] = c_override_rate;
    c_obj["G_action"] = c_G_act;
    c_obj["G_exec"] = c_G_exec;
    c_obj["G_target"] = c_G_tgt;
    c_obj["neg_override_rate"] = c_neg_rate;
    c_obj["mean_rollout_steps"] = c_mean_steps;
    c_obj["mean_rollout_nn_evals"] = c_mean_evals;
    categories_json.push_back(std::move(c_obj));
  }
  root_obj["categories"] = std::move(categories_json);

  std::filesystem::path op(output_path);
  if (!op.parent_path().empty()) std::filesystem::create_directories(op.parent_path());
  std::ofstream ofs(output_path);
  if (ofs.is_open()) {
    ofs << json::ToString(root_obj, true) << "\n";
    ofs.flush();
    ofs.close();
    std::cout << "[R755-GATE1] Wrote Gate 1 audit artifact to " << output_path << "\n";
  } else {
    std::cerr << "[R755-GATE1] ERROR: Could not open " << output_path << " for writing!\n";
  }
  std::cout << "[R755-GATE1] Summary: G_action=" << G_action
            << " [" << boot_G_action[250] << ", " << boot_G_action[9750] << "]"
            << " (LCB95=" << boot_G_action[500] << ")"
            << ", G_exec=" << G_exec
            << " (LCB95=" << boot_G_exec[500] << ")"
            << ", G_target=" << G_target
            << " (LCB95=" << boot_G_target[500] << ")"
            << ", neg_override_rate=" << neg_override_rate
            << ", mean_delta_p=" << mean_delta_p << std::endl;
  std::exit(0);
}



#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH

}  // namespace open_spiel::ppo_train_internal
