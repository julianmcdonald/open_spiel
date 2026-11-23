#include "dune_ppo_train_sim_internal.h"

namespace open_spiel::ppo_train_internal {

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

namespace {

void InitSimContext(
    uint64_t master, uint64_t episode_id, const Game& game,
    std::shared_ptr<IGameEvaluator> evaluator, int64_t obs_size,
    std::vector<PpoTransition>* trajectory,
    std::atomic<uint64_t>* total_env_steps, float reward_lambda,
    WorkerStats* local_stats,
    const std::vector<std::shared_ptr<IGameEvaluator>>& opponent_evaluators,
    std::vector<open_spiel::dune_imperium::SearchSnapshot>* local_snapshots,
    uint64_t* decision_count, std::mt19937_64* reservoir_rng,
    std::vector<open_spiel::SearchTrainingExample>* r755_aux_examples,
    PpoSimContext* ctx) {
  ctx->master = master;
  ctx->episode_id = episode_id;
  ctx->game = &game;
  ctx->evaluator = std::move(evaluator);
  ctx->obs_size = obs_size;
  ctx->trajectory = trajectory;
  ctx->total_env_steps = total_env_steps;
  ctx->reward_lambda = reward_lambda;
  ctx->local_stats = local_stats;
  ctx->opponent_evaluators = &opponent_evaluators;
  ctx->local_snapshots = local_snapshots;
  ctx->decision_count = decision_count;
  ctx->reservoir_rng = reservoir_rng;
  ctx->r755_aux_examples = r755_aux_examples;

  ctx->single_learner = absl::GetFlag(FLAGS_single_learner_training);
  const double pool_game_fraction = absl::GetFlag(FLAGS_pool_game_fraction);
  const int pool_opponents_per_game = absl::GetFlag(FLAGS_pool_opponents_per_game);
  if (ctx->single_learner) {
    ctx->seat_assignment = dune_opponent_pool::ComputeSeatAssignment(
        master, episode_id, opponent_evaluators.size());
  } else if (pool_game_fraction > 0.0 && pool_opponents_per_game > 0) {
    ctx->seat_assignment = dune_opponent_pool::ComputeMixedPoolSeatAssignment(
        master, episode_id, opponent_evaluators.size(), pool_game_fraction,
        pool_opponents_per_game);
  } else {
    ctx->seat_assignment = dune_opponent_pool::SeatAssignment{};
  }
  ctx->learner_seat = ctx->single_learner ? ctx->seat_assignment.learner_seat : -1;

  ctx->chance_rng = dune_seed::MakeRng64(
      dune_seed::DeriveSeed(master, dune_seed::kDomainTrain, episode_id,
                            dune_seed::kStreamChance));
  for (int p = 0; p < 4; ++p) {
    uint64_t seed = dune_seed::DeriveSeed(
        master, dune_seed::kDomainTrain, episode_id,
        dune_seed::kStreamPolicyPlayer0 + p);
    ctx->policy_rng[p] = dune_seed::MakeRng64(seed);
  }

  if (absl::GetFlag(FLAGS_purchase_exploration)) {
    uint64_t seed = dune_seed::DeriveSeed(
        master, dune_seed::kDomainTrain, episode_id,
        dune_seed::kStreamPurchaseExploration);
    ctx->explore_rng = dune_seed::MakeRng64(seed);
    std::uniform_real_distribution<double> p_dist(0.0, 1.0);
    if (p_dist(ctx->explore_rng) < 0.10) {
      ctx->do_purchase_exploration = true;
      if (ctx->single_learner) {
        ctx->purchase_exploration_seat = ctx->learner_seat;
      } else {
        std::uniform_int_distribution<int> seat_dist(0, game.NumPlayers() - 1);
        ctx->purchase_exploration_seat = seat_dist(ctx->explore_rng);
      }
      std::vector<int> rounds_pool = {1, 2, 3, 4, 5, 6};
      std::shuffle(rounds_pool.begin(), rounds_pool.end(), ctx->explore_rng);
      ctx->forced_round_1 = rounds_pool[0];
      ctx->forced_round_2 = rounds_pool[1];
    }
  }

  ctx->designated_explore_seat =
      ctx->single_learner ? ctx->learner_seat : static_cast<int>(episode_id % 4);
}

void InitEpisodeState(PpoSimContext* ctx) {
  const Game& game = *ctx->game;
  ctx->r755_hybrid = absl::GetFlag(FLAGS_r755_hybrid_mode);
  if (ctx->r755_hybrid) {
    auto batched_eval = std::dynamic_pointer_cast<open_spiel::BatchedEvaluator>(ctx->evaluator);
    if (batched_eval != nullptr) {
      ctx->r755_search_evals.push_back(std::make_shared<open_spiel::BatchedNNEvaluator>(
          batched_eval, static_cast<float>(absl::GetFlag(FLAGS_logit_cap)),
          dune_imperium::MarketAppendixMode::kNone,
          /*forward_active_rollouts=*/false));
    }
  }

  ctx->state = game.NewInitialState();
  ctx->provides_info_state_tensor = game.GetType().provides_information_state_tensor;
  ctx->provides_observations_tensor = game.GetType().provides_observation_tensor;
  ctx->dune_state = dynamic_cast<const dune_imperium::DuneImperiumState*>(ctx->state.get());

  const int num_players = game.NumPlayers();
  ctx->current_tleilaxu.assign(num_players, 0);
  ctx->had_swordmaster.assign(num_players, false);
  ctx->prev_conflict_vp.assign(num_players, 0);
  ctx->prev_total_vp.assign(num_players, 0);
  ctx->prev_hand_sb.assign(num_players, 0);
  ctx->prev_hand_sh.assign(num_players, 0);
  if (ctx->dune_state != nullptr) {
    for (int p = 0; p < num_players; ++p) {
      ctx->current_tleilaxu[p] = ctx->dune_state->GetTleilaxuTrackForTesting(p);
      ctx->had_swordmaster[p] = ctx->dune_state->HasSwordmaster(p);
      ctx->prev_conflict_vp[p] = ctx->dune_state->ConflictVpDelta(p);
      ctx->prev_total_vp[p] = ctx->dune_state->GetPlayerVp(p);
      const auto& hand = ctx->dune_state->GetPlayerHandForTesting(p);
      for (int c : hand) {
        if (c == dune_imperium::kCardScientificBreakthrough) ++ctx->prev_hand_sb[p];
        if (c == dune_imperium::kCardStitchedHorror) ++ctx->prev_hand_sh[p];
      }
    }
  }

  ctx->last_transition_index.assign(num_players, -1);
  ctx->pre_combat_strength.assign(num_players, 0);

  ctx->logit_cap = static_cast<float>(absl::GetFlag(FLAGS_logit_cap));
  ctx->shaped_weight = absl::GetFlag(FLAGS_shaped_reward_weight);
  ctx->tleilaxu_breadcrumb_weight = absl::GetFlag(FLAGS_tleilaxu_breadcrumb_weight);
  ctx->tleilaxu_level7_breadcrumb_weight =
      absl::GetFlag(FLAGS_tleilaxu_level7_breadcrumb_weight);
  ctx->specimen_exchange_penalty = absl::GetFlag(FLAGS_specimen_exchange_penalty);
  ctx->family_atomics_penalty = absl::GetFlag(FLAGS_family_atomics_penalty);
  ctx->plot_intrigue_penalty = absl::GetFlag(FLAGS_plot_intrigue_penalty);
  ctx->plot_intrigue_exemption_threshold =
      absl::GetFlag(FLAGS_plot_intrigue_exemption_threshold);

  // PWO-5 section 14.1b: hoisted per episode so the canary costs one bool test when off.
  ctx->emit_canary_columns =
      (ctx->local_stats != nullptr) && absl::GetFlag(FLAGS_emit_canary_columns);
}

void FinalizeEpisodeReturnsAndGae(PpoSimContext* ctx) {
  const Game& game = *ctx->game;
  std::vector<PpoTransition>* trajectory = ctx->trajectory;
  WorkerStats* local_stats = ctx->local_stats;

  const std::vector<double> terminal_returns = open_spiel::ComputeTerminalReturns(
      *ctx->state,
      absl::GetFlag(FLAGS_terminal_reward_mode),
      absl::GetFlag(FLAGS_round7_speed_bonus),
      absl::GetFlag(FLAGS_win_speed_bonus_per_round));
  double gamma = absl::GetFlag(FLAGS_train_value_only) ? 1.0 : absl::GetFlag(FLAGS_gamma);
  double gae_lambda =
      absl::GetFlag(FLAGS_train_value_only) ? 1.0 : absl::GetFlag(FLAGS_gae_lambda);
  float reward_scale = static_cast<float>(
      std::max(1e-6, absl::GetFlag(FLAGS_reward_scale)));
  const float max_terminal =
      (absl::GetFlag(FLAGS_terminal_reward_mode) == "first_place" ? 3.0f : 2.25f) +
      static_cast<float>(
          std::max(0.0, absl::GetFlag(FLAGS_round7_speed_bonus)) +
          4.0 * std::max(0.0, absl::GetFlag(FLAGS_win_speed_bonus_per_round)));
  const float effective_scale =
      (max_terminal > reward_scale) ? std::max(reward_scale, max_terminal + 0.05f)
                                    : reward_scale;
  std::vector<float> last_value(game.NumPlayers(), 0.0f);
  std::vector<float> last_gae(game.NumPlayers(), 0.0f);
  std::vector<bool> seen_last_action(game.NumPlayers(), false);
  std::vector<float> running_mc_return(game.NumPlayers(), 0.0f);

  for (auto it = trajectory->rbegin(); it != trajectory->rend(); ++it) {
    if (it->player_id < 0 || it->player_id >= game.NumPlayers()) continue;
    int p = it->player_id;
    float reward = 0.0f;
    if (absl::GetFlag(FLAGS_train_value_only)) {
      if (!seen_last_action[p]) {
        reward = static_cast<float>(terminal_returns[p]);
        seen_last_action[p] = true;
      }
    } else {
      reward = it->reward;
      if (!seen_last_action[p]) {
        reward += static_cast<float>(terminal_returns[p]);
        seen_last_action[p] = true;
      }
    }
    reward = std::clamp(reward / effective_scale, -1.0f, 1.0f);
    it->reward = reward;

    running_mc_return[p] += reward;
    it->mc_return = running_mc_return[p];

    float delta = reward + static_cast<float>(gamma) * last_value[p] - it->value;
    float advantage =
        delta + static_cast<float>(gamma * gae_lambda) * last_gae[p];
    it->advantage = advantage;
    it->return_value = advantage + it->value;
    last_value[p] = it->value;
    last_gae[p] = advantage;
    it->gae_advantage = advantage;

    if (it->targeted_buy_forced &&
        absl::GetFlag(FLAGS_targeted_buy_policy_advantage) == "full_return") {
      it->advantage = it->mc_return - it->value;
    }

    if (local_stats != nullptr && it->targeted_buy_card > 0) {
      int c_idx = (it->targeted_buy_card ==
                   open_spiel::dune_targeted_buy::kScientificBreakthroughTleilaxuId)
                      ? 0
                      : 1;
      int r_idx = std::clamp(it->targeted_buy_round, 1, 10);
      if (it->targeted_buy_forced) {
        local_stats->targeted_buy_telemetry.forced_buys[c_idx][r_idx]++;
        local_stats->targeted_buy_telemetry.forced_sum_adv[c_idx][r_idx] += it->advantage;
      } else {
        local_stats->targeted_buy_telemetry.unforced_sum_adv[c_idx][r_idx] += advantage;
      }
    }
  }

  std::vector<std::vector<size_t>> player_trans_indices(game.NumPlayers());
  for (size_t idx = 0; idx < trajectory->size(); ++idx) {
    int p = (*trajectory)[idx].player_id;
    if (p >= 0 && p < game.NumPlayers()) {
      player_trans_indices[p].push_back(idx);
    }
  }
  for (int p = 0; p < game.NumPlayers(); ++p) {
    const size_t n_p = player_trans_indices[p].size();
    if (n_p == 0) continue;
    for (size_t i = 0; i < n_p; ++i) {
      size_t idx = player_trans_indices[p][i];
      int third = 0;
      if (i * 3 < n_p) {
        third = 0;
      } else if (i * 3 < 2 * n_p) {
        third = 1;
      } else {
        third = 2;
      }
      (*trajectory)[idx].trajectory_third = third;
    }
  }

  if (local_stats != nullptr) {
    for (int p = 0; p < game.NumPlayers(); ++p) {
      const size_t n_p = player_trans_indices[p].size();
      if (n_p < 2) continue;
      for (size_t i = 0; i + 1 < n_p; ++i) {
        const auto& t_buy = (*trajectory)[player_trans_indices[p][i]];
        if (t_buy.round >= 1 && t_buy.round <= 6 && !t_buy.mask_policy_loss &&
            open_spiel::dune_imperium::IsCardOrTechAcquisitionAction(t_buy.action)) {
          const auto& t_after = (*trajectory)[player_trans_indices[p][i + 1]];
          double i_val = 4.0 * (t_buy.mc_return - t_buy.value);
          double ii_val = 4.0 * (t_after.mc_return - t_after.value);
          double iii_val = 4.0 * (t_buy.mc_return - (t_buy.value + t_buy.gae_advantage));

          int r_bucket = (t_buy.round <= 2) ? 0 : ((t_buy.round <= 4) ? 1 : 2);
          double prior = std::exp(static_cast<double>(t_buy.old_log_prob));
          int p_bucket = (prior < 0.01) ? 0 : ((prior < 0.05) ? 1 : 2);
          int b_idx = r_bucket * 3 + p_bucket;

          local_stats->r755_cat3_i_sum[b_idx] += i_val;
          local_stats->r755_cat3_ii_sum[b_idx] += ii_val;
          local_stats->r755_cat3_iii_sum[b_idx] += iii_val;
          local_stats->r755_cat3_bucket_count[b_idx]++;
        }
      }
    }
  }

  for (auto& cf : ctx->pending_cf) {
    for (int repeat = 0; repeat < cf.replay_weight; ++repeat) {
      if (repeat + 1 == cf.replay_weight) {
        trajectory->push_back(std::move(cf.transition));
      } else {
        trajectory->push_back(cf.transition);
      }
    }
  }
}

}  // namespace

int PpoSimulation(uint64_t master, uint64_t episode_id, const Game& game,
                  std::shared_ptr<IGameEvaluator> evaluator, int64_t obs_size,
                  std::vector<PpoTransition>* trajectory,
                  std::atomic<uint64_t>* total_env_steps,
                  float reward_lambda,
                  WorkerStats* local_stats,
                  const std::vector<std::shared_ptr<IGameEvaluator>>& opponent_evaluators,
                  std::vector<open_spiel::dune_imperium::SearchSnapshot>* local_snapshots,
                  uint64_t* decision_count,
                  std::mt19937_64* reservoir_rng,
                  std::vector<open_spiel::SearchTrainingExample>* r755_aux_examples) {
  PpoSimContext ctx;
  InitSimContext(master, episode_id, game, std::move(evaluator), obs_size,
                 trajectory, total_env_steps, reward_lambda, local_stats,
                 opponent_evaluators, local_snapshots, decision_count,
                 reservoir_rng, r755_aux_examples, &ctx);

  while (true) {
    InitEpisodeState(&ctx);
    torch::NoGradGuard no_grad;

    while (!ctx.state->IsTerminal()) {
      ++ctx.game_length;
      if (ctx.game_length > 5000) {
        std::cerr << "Possible infinite loop detected. State:\n"
                  << ctx.state->ToString() << "\n";
        std::abort();
      }

      if (ctx.state->IsChanceNode()) {
        std::vector<std::pair<Action, double>> outcomes = ctx.state->ChanceOutcomes();
        Action action = game.GetType().chance_mode == GameType::ChanceMode::kSampledStochastic
                            ? outcomes.front().first
                            : SampleAction(outcomes, ctx.chance_rng).first;
        ctx.state->ApplyAction(action);
        CheckHandDraws(&ctx);
        ctx.r765_saw_chance_since_root.fill(true);
        if (ctx.r755_hybrid) {
          for (auto& t : ctx.r755_turn_trees) t.Reset();
        }
        continue;
      }

      if (ctx.state->CurrentPlayer() == kSimultaneousPlayerId) {
        std::vector<Action> joint_action;
        for (int p = 0; p < game.NumPlayers(); ++p) {
          std::vector<Action> actions = ctx.state->LegalActions(p);
          if (actions.empty()) {
            joint_action.push_back(0);
          } else {
            std::uniform_int_distribution<int> dis(0, actions.size() - 1);
            joint_action.push_back(actions[dis(ctx.policy_rng[p])]);
          }
        }
        ctx.state->ApplyActions(joint_action);
        CheckHandDraws(&ctx);
        continue;
      }

      if (absl::GetFlag(FLAGS_train_value_only)) {
        if (!prohibited_histories.empty() &&
            prohibited_histories.count(ctx.state->History())) {
          SpielFatalError(
              "CRITICAL PROTECTION BOUNDARY FAILURE: Attempted to process or train "
              "on a prohibited official-corpus state!");
        }
      }

      SimStepDecision step;
      step.current_player = ctx.state->CurrentPlayer();
      step.actions = ctx.state->LegalActions();
      if (step.actions.empty()) {
        std::cerr << "Non-terminal state has no legal actions. State:\n"
                  << ctx.state->ToString() << "\n";
        std::abort();
      }

      step.is_learner = (step.current_player >= 0 && step.current_player < 4) &&
                        ctx.seat_assignment.is_learner_seat[step.current_player];

      if (ctx.last_player >= 0 && ctx.last_player < 4 &&
          step.current_player != ctx.last_player) {
        ctx.r765_pending_joint[ctx.last_player].Clear();
      }
      if (!step.is_learner && step.current_player >= 0 && step.current_player < 4) {
        ctx.r765_pending_joint[step.current_player].Clear();
      }

      if (ctx.r755_hybrid) {
        ++ctx.r755_decision_index;
        if (step.current_player != ctx.r755_last_turn_player ||
            ctx.r755_last_action == dune_imperium::kActionEndTurn) {
          if (step.current_player >= 0 && step.current_player < 4) {
            ctx.r755_space_placed_this_turn[step.current_player] = false;
            ctx.r755_turn_trees[step.current_player].Reset();
            uint64_t coin_seed = dune_seed::DeriveSeed(
                master, dune_seed::kDomainTrain, episode_id,
                ctx.r755_decision_index, 0x52373535ULL);
            std::mt19937_64 coin_rng(coin_seed);
            const double turn_coin_prob =
                (absl::GetFlag(FLAGS_r765_agent_search_prob) >= 0.0)
                    ? absl::GetFlag(FLAGS_r765_agent_search_prob)
                    : absl::GetFlag(FLAGS_r755_search_prob);
            ctx.r755_turn_coin_active[step.current_player] =
                (std::uniform_real_distribution<double>(0.0, 1.0)(coin_rng) <
                 turn_coin_prob);
          }
        }
      }

      EvaluateAndSelectStepAction(&ctx, &step);
      TrackTwoCardDecisions(&ctx, step);
      ValidateParityCapturePostStep(&ctx, step);
      SampleCounterfactualSuccessor(&ctx, step);
      CapturePreActionTrackingAndUpdateTrees(&ctx, &step);

      ctx.state->ApplyAction(step.action);
      ctx.last_player = step.current_player;
      ctx.last_action = step.action;
      ctx.total_env_steps->fetch_add(1, std::memory_order_relaxed);
      CheckHandDraws(&ctx);

      RecordPostActionHighStakesAndSwordmaster(&ctx, &step);
      RecordLearnerTransition(&ctx, &step);
      ApplyStepTrackingAndShaping(&ctx, step);
    }

    FinalizeEpisodeReturnsAndGae(&ctx);
    return ctx.game_length;
  }
}


void SampleCounterfactualSuccessor(PpoSimContext* ctx, const SimStepDecision& step) {
  const Player current_player = step.current_player;
  if (!absl::GetFlag(FLAGS_sample_counterfactual_states) || !step.is_learner ||
      ctx->counterfactual_samples >= absl::GetFlag(FLAGS_counterfactual_samples_per_game) ||
      ClassifyDuneDecisionRole(*ctx->state, current_player, false) !=
          DuneDecisionRole::kAgentPrimary) {
    return;
  }

  // Clone independent RNGs to prevent perturbing the live trajectory.
  std::mt19937_64 cf_chance_rng = ctx->chance_rng;
  std::mt19937_64 cf_policy_rng[4];
  for (int p = 0; p < 4; ++p) {
    cf_policy_rng[p] = ctx->policy_rng[p];
  }

  std::vector<Action> alternate_actions;
  alternate_actions.reserve(step.actions.size());
  for (Action candidate : step.actions) {
    if (candidate != step.action) alternate_actions.push_back(candidate);
  }
  if (alternate_actions.empty()) return;

  std::uniform_int_distribution<size_t> alternate_distribution(
      0, alternate_actions.size() - 1);
  Action alt_act = alternate_actions[alternate_distribution(cf_policy_rng[current_player])];

  auto succ_state = ctx->state->Clone();
  succ_state->ApplyAction(alt_act);

  while (succ_state->IsChanceNode()) {
    auto outcomes = succ_state->ChanceOutcomes();
    if (outcomes.empty()) break;
    Action chance_act =
        ctx->game->GetType().chance_mode == GameType::ChanceMode::kSampledStochastic
            ? outcomes.front().first
            : SampleAction(outcomes, cf_chance_rng).first;
    succ_state->ApplyAction(chance_act);
  }

  std::vector<float> succ_obs(ctx->obs_size, 0.0f);
  bool valid = false;
  float succ_val = 0.0f;
  float target_return = 0.0f;
  if (!succ_state->IsTerminal()) {
    const auto* d_succ =
        dynamic_cast<const dune_imperium::DuneImperiumState*>(succ_state.get());
    if (d_succ != nullptr) {
      d_succ->InformationStateTensorWithAppendix(
          current_player, ctx->evaluator->MarketMode(), absl::MakeSpan(succ_obs),
          ctx->evaluator->DeckPoolSemantics());
      valid = true;
    } else if (ctx->provides_info_state_tensor) {
      succ_state->InformationStateTensor(current_player, absl::MakeSpan(succ_obs));
      valid = true;
    } else if (ctx->provides_observations_tensor) {
      succ_state->ObservationTensor(current_player, absl::MakeSpan(succ_obs));
      valid = true;
    }
    if (valid) {
      EvalResult succ_result = ctx->evaluator->Evaluate(succ_obs);
      succ_val = succ_result.value;

      auto roll_state = succ_state->Clone();
      while (!roll_state->IsTerminal()) {
        if (roll_state->IsChanceNode()) {
          auto outcomes = roll_state->ChanceOutcomes();
          if (outcomes.empty()) break;
          Action choice =
              ctx->game->GetType().chance_mode == GameType::ChanceMode::kSampledStochastic
                  ? outcomes.front().first
                  : SampleAction(outcomes, cf_chance_rng).first;
          roll_state->ApplyAction(choice);
          continue;
        }

        if (roll_state->CurrentPlayer() == kSimultaneousPlayerId) {
          std::vector<Action> joint_action;
          for (int p = 0; p < ctx->game->NumPlayers(); ++p) {
            std::vector<Action> p_actions = roll_state->LegalActions(p);
            if (p_actions.empty()) {
              joint_action.push_back(0);
            } else {
              std::uniform_int_distribution<int> dis(0, p_actions.size() - 1);
              joint_action.push_back(p_actions[dis(cf_policy_rng[p])]);
            }
          }
          roll_state->ApplyActions(joint_action);
          continue;
        }

        Player curr_p = roll_state->CurrentPlayer();
        std::vector<Action> curr_acts = roll_state->LegalActions();
        std::vector<float> curr_obs(ctx->obs_size, 0.0f);
        const auto* d_roll =
            dynamic_cast<const dune_imperium::DuneImperiumState*>(roll_state.get());
        if (d_roll != nullptr) {
          d_roll->InformationStateTensorWithAppendix(
              curr_p, ctx->evaluator->MarketMode(), absl::MakeSpan(curr_obs),
              ctx->evaluator->DeckPoolSemantics());
        } else if (ctx->provides_info_state_tensor) {
          roll_state->InformationStateTensor(curr_p, absl::MakeSpan(curr_obs));
        } else {
          roll_state->ObservationTensor(curr_p, absl::MakeSpan(curr_obs));
        }
        EvalResult roll_res = ctx->evaluator->Evaluate(curr_obs);
        CenterAndCapLegalLogits(roll_res.logits, curr_acts, ctx->logit_cap);
        Action roll_act =
            SamplePolicyAction(&cf_policy_rng[curr_p], roll_res.logits, curr_acts).first;
        roll_state->ApplyAction(roll_act);
      }
      target_return = static_cast<float>(roll_state->Returns()[current_player]) / 4.0f;
    }
  } else {
    succ_val = static_cast<float>(succ_state->Returns()[current_player]) / 4.0f;
    target_return = succ_val;
    valid = true;
  }

  if (valid) {
    if (absl::GetFlag(FLAGS_train_value_only)) {
      if (!prohibited_histories.empty() &&
          prohibited_histories.count(succ_state->History())) {
        SpielFatalError(
            "CRITICAL PROTECTION BOUNDARY FAILURE: Attempted to process or train on a "
            "prohibited official-corpus state via counterfactual generation!");
      }
    }
    PpoTransition cf_trans;
    cf_trans.state = std::move(succ_obs);
    cf_trans.legal_actions =
        succ_state->IsTerminal() ? std::vector<Action>{} : succ_state->LegalActions();
    cf_trans.action = kInvalidAction;
    cf_trans.old_log_prob = 0.0f;
    cf_trans.reward = 0.0f;
    cf_trans.value = succ_val;
    cf_trans.advantage = target_return - succ_val;
    cf_trans.return_value = target_return;
    cf_trans.player_id = current_player;
    cf_trans.episode_id = ctx->episode_id;

    CounterfactualPending pending;
    pending.replay_weight = absl::GetFlag(FLAGS_counterfactual_replay_weight);
    pending.transition = std::move(cf_trans);
    ctx->pending_cf.push_back(std::move(pending));
    ++ctx->counterfactual_samples;
  }
}

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH

}  // namespace open_spiel::ppo_train_internal
