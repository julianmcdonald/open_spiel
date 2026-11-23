#include "dune_ppo_train_sim_internal.h"

namespace open_spiel::ppo_train_internal {

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

namespace {

void RecordCardPlay(PpoSimContext* ctx, Player current_player, int card) {
  if (card == dune_imperium::kCardScientificBreakthrough) {
    ctx->local_stats->card11_metrics.plays++;
    if (ctx->dune_state != nullptr &&
        ctx->dune_state->HasSecondGeneticMarker(current_player)) {
      ctx->local_stats->card11_metrics.plays_with_second_marker++;
    } else {
      ctx->local_stats->card11_metrics.plays_without_second_marker++;
    }
  }
  if (card == dune_imperium::kCardStitchedHorror) {
    ctx->local_stats->card13_metrics.plays++;
  }
}

}  // namespace

void CheckHandDraws(PpoSimContext* ctx) {
  if (ctx->dune_state == nullptr || ctx->local_stats == nullptr) return;
  for (int p = 0; p < ctx->game->NumPlayers(); ++p) {
    int count_sb = 0;
    int count_sh = 0;
    const auto& hand = ctx->dune_state->GetPlayerHandForTesting(p);
    for (int c : hand) {
      if (c == dune_imperium::kCardScientificBreakthrough) ++count_sb;
      if (c == dune_imperium::kCardStitchedHorror) ++count_sh;
    }
    if (count_sb > ctx->prev_hand_sb[p]) {
      ctx->local_stats->card11_metrics.draws_into_hand += (count_sb - ctx->prev_hand_sb[p]);
    }
    if (count_sh > ctx->prev_hand_sh[p]) {
      ctx->local_stats->card13_metrics.draws_into_hand += (count_sh - ctx->prev_hand_sh[p]);
    }
    ctx->prev_hand_sb[p] = count_sb;
    ctx->prev_hand_sh[p] = count_sh;
  }
}

void TrackTwoCardDecisions(PpoSimContext* ctx, const SimStepDecision& step) {
  // Two-card tracking: Scientific Breakthrough (Tleilaxu 11 / Card 118) and
  // Stitched Horror (Tleilaxu 13 / Card 120).
  if (ctx->dune_state == nullptr || ctx->local_stats == nullptr) return;

  const Player current_player = step.current_player;
  const Action action = step.action;
  const std::vector<Action>& actions = step.actions;
  const std::vector<double>& legal_probabilities = step.legal_probabilities;

  const auto& tleilaxu_row = ctx->dune_state->GetTleilaxuRowForTesting();
  int row_card_0 = (tleilaxu_row.size() > 0) ? tleilaxu_row[0] : dune_imperium::kInvalidCard;
  int row_card_1 = (tleilaxu_row.size() > 1) ? tleilaxu_row[1] : dune_imperium::kInvalidCard;

  // kActionTleilaxuAcquire0 = 90 (Reclaimed Forces), +1 = Slot 0, +2 = Slot 1.
  const Action action_buy_0 = dune_imperium::kActionTleilaxuAcquire0 + 1;
  const Action action_buy_1 = dune_imperium::kActionTleilaxuAcquire0 + 2;

  int buy_idx_0 = -1;
  int buy_idx_1 = -1;
  for (size_t idx = 0; idx < actions.size(); ++idx) {
    if (actions[idx] == action_buy_0) buy_idx_0 = static_cast<int>(idx);
    if (actions[idx] == action_buy_1) buy_idx_1 = static_cast<int>(idx);
  }

  auto process_card_opp = [&](int target_tleilaxu_id, int slot, int action_idx, Action buy_action) {
    auto& card_stats = (target_tleilaxu_id == 11)
                           ? ctx->local_stats->card11_metrics
                           : ctx->local_stats->card13_metrics;
    card_stats.legal_opportunities++;
    if (slot == 0) card_stats.legal_opps_slot0++;
    else card_stats.legal_opps_slot1++;

    if (action_idx >= 0 && static_cast<size_t>(action_idx) < legal_probabilities.size()) {
      card_stats.sum_purchase_probs += legal_probabilities[action_idx];
    }
    if (action == buy_action) {
      card_stats.purchases_taken++;
    }
  };

  if (buy_idx_0 >= 0) {
    if (row_card_0 == 11) process_card_opp(11, 0, buy_idx_0, action_buy_0);
    else if (row_card_0 == 13) process_card_opp(13, 0, buy_idx_0, action_buy_0);
  }
  if (buy_idx_1 >= 0) {
    if (row_card_1 == 11) process_card_opp(11, 1, buy_idx_1, action_buy_1);
    else if (row_card_1 == 13) process_card_opp(13, 1, buy_idx_1, action_buy_1);
  }

  if (action >= dune_imperium::kActionSelectAgentCard0 &&
      action < dune_imperium::kActionSelectAgentCard0 + 200) {
    RecordCardPlay(ctx, current_player, action - dune_imperium::kActionSelectAgentCard0);
  }
  if (action >= dune_imperium::kActionSelectGraftPartner0 &&
      action < dune_imperium::kActionSelectGraftPartner0 + 200) {
    RecordCardPlay(ctx, current_player, action - dune_imperium::kActionSelectGraftPartner0);
  }

  for (Action a : actions) {
    if (a == dune_imperium::kActionAgentRewardScientificBreakthroughTrash) {
      ctx->local_stats->card11_metrics.trash_for_vp_offered++;
      break;
    }
  }

  if (action == dune_imperium::kActionAgentRewardScientificBreakthroughTrash) {
    ctx->local_stats->card11_metrics.trash_for_vp_resolved++;
    ctx->local_stats->card11_metrics.special_effects_resolved++;
  }
  if (action == dune_imperium::kActionAgentRewardScientificBreakthroughSkip) {
    ctx->local_stats->card11_metrics.trash_for_vp_skipped++;
  }

  if (action == dune_imperium::kActionStitchedHorror01) {
    ctx->local_stats->card13_metrics.choice_01_water_troop++;
  } else if (action == dune_imperium::kActionStitchedHorror02) {
    ctx->local_stats->card13_metrics.choice_02_water_trash++;
    ctx->pending_horror_trash[current_player] = true;
  } else if (action == dune_imperium::kActionStitchedHorror03) {
    ctx->local_stats->card13_metrics.choice_03_water_scarab++;
  } else if (action == dune_imperium::kActionStitchedHorror12) {
    ctx->local_stats->card13_metrics.choice_12_troop_trash++;
    ctx->pending_horror_trash[current_player] = true;
  } else if (action == dune_imperium::kActionStitchedHorror13) {
    ctx->local_stats->card13_metrics.choice_13_troop_scarab++;
  } else if (action == dune_imperium::kActionStitchedHorror23) {
    ctx->local_stats->card13_metrics.choice_23_scarab_trash++;
    ctx->pending_horror_trash[current_player] = true;
  }

  // Strictly track Stitched Horror trash resolution ONLY when pending_horror_trash is true.
  if (ctx->pending_horror_trash[current_player]) {
    if (action >= dune_imperium::kActionTrashInPlayCard0 &&
        action < dune_imperium::kActionTrashInPlayCard0 + 200) {
      ctx->local_stats->card13_metrics.in_play_trash_resolved++;
      ctx->local_stats->card13_metrics.special_effects_resolved++;
      ctx->pending_horror_trash[current_player] = false;
    } else if (action >= dune_imperium::kActionTrashHandCard0 &&
               action < dune_imperium::kActionTrashHandCard0 + 200) {
      ctx->local_stats->card13_metrics.hand_trash_resolved++;
      ctx->local_stats->card13_metrics.special_effects_resolved++;
      ctx->pending_horror_trash[current_player] = false;
    } else if (action >= dune_imperium::kActionTrashDiscardCard0 &&
               action < dune_imperium::kActionTrashDiscardCard0 + 200) {
      ctx->local_stats->card13_metrics.discard_trash_resolved++;
      ctx->local_stats->card13_metrics.special_effects_resolved++;
      ctx->pending_horror_trash[current_player] = false;
    } else if (action == dune_imperium::kActionTrashSkip) {
      ctx->local_stats->card13_metrics.trash_skipped++;
      ctx->pending_horror_trash[current_player] = false;
    } else if (action != dune_imperium::kActionStitchedHorror02 &&
               action != dune_imperium::kActionStitchedHorror12 &&
               action != dune_imperium::kActionStitchedHorror23) {
      ctx->pending_horror_trash[current_player] = false;
    }
  }
}

void ValidateParityCapturePostStep(PpoSimContext* ctx, const SimStepDecision& step) {
  if (!step.parity_capture) return;
  const std::vector<int64_t> legal_ids(step.actions.begin(), step.actions.end());
  std::string validation_error;
  if (ValidatePpoParityBehaviorCaptureWidthsAndChoice(
          legal_ids, step.action, step.parity_raw_legal_logits,
          step.parity_behavior_log_probs, &validation_error)) {
    ++ctx->local_stats->parity_chosen_once;
  } else {
    ctx->local_stats->parity_capture_errors.push_back(absl::StrFormat(
        "episode=%llu decision=%lld: %s",
        static_cast<unsigned long long>(ctx->episode_id),
        static_cast<long long>(ctx->trajectory->size()), validation_error));
  }
}

void CapturePreActionTrackingAndUpdateTrees(PpoSimContext* ctx, SimStepDecision* step) {
  const Player current_player = step->current_player;
  const Action action = step->action;

  step->is_atomics_action = (action == dune_imperium::kActionFamilyAtomics);
  step->atomics_exempt = false;
  step->is_plot_intrigue_action = open_spiel::IsPlotIntrigueAction(action);
  step->plot_intrigue_exempt = false;
  if (ctx->dune_state != nullptr && current_player >= 0 &&
      current_player < ctx->game->NumPlayers()) {
    ctx->pre_action_phase = ctx->dune_state->phase();
    ctx->pre_combat_strength[current_player] =
        dune_imperium::CombatStrength(*ctx->dune_state, current_player);
    if (step->is_atomics_action) {
      step->atomics_exempt =
          ctx->dune_state->IsPlayerInRevealTurn(current_player) &&
          (ctx->dune_state->PlayerPersuasion(current_player) > 0);
    }
    if (step->is_plot_intrigue_action) {
      int hand_size = static_cast<int>(
          ctx->dune_state->GetIntrigueHandForTesting(current_player).size());
      step->plot_intrigue_exempt = (hand_size >= ctx->plot_intrigue_exemption_threshold);
    }
  }

  step->pre_action_privileged_appendix.clear();
  if (absl::GetFlag(FLAGS_privileged_critic) && step->is_learner &&
      ctx->dune_state != nullptr) {
    const auto central = ctx->dune_state->VrpoCentralCriticTensor(current_player);
    SPIEL_CHECK_EQ(central.size(), 9012);
    constexpr size_t kAppendixOffset = 5580;
    constexpr size_t kAppendixSize = 3432;
    step->pre_action_privileged_appendix.resize(kAppendixSize);
    for (size_t k = 0; k < kAppendixSize; ++k) {
      step->pre_action_privileged_appendix[k] =
          open_spiel::FloatToFp16(central[kAppendixOffset + k]);
    }
  }

  step->pre_action_is_high_stakes = false;
  if (step->is_learner) {
    step->pre_action_is_high_stakes =
        dune_imperium::IsHighStakesStrategicState(*ctx->state, current_player);
  }

  if (ctx->r755_hybrid && current_player >= 0 && current_player < 4) {
    if (action == dune_imperium::kActionEndTurn ||
        action == dune_imperium::kActionReveal) {
      ctx->r765_pending_joint[current_player].Clear();
    }
    if (action == dune_imperium::kActionEndTurn) {
      ctx->r755_turn_trees[current_player].Reset();
    } else {
      ctx->r755_turn_trees[current_player].Advance(action);
    }
    if (action >= dune_imperium::kActionAgentSpaceConspire &&
        action <= dune_imperium::kActionAgentSpaceSietchTabr) {
      ctx->r755_space_placed_this_turn[current_player] = true;
    }
    ctx->r755_last_turn_player = current_player;
    ctx->r755_last_action = action;
  }
}

void RecordPostActionHighStakesAndSwordmaster(PpoSimContext* ctx, SimStepDecision* step) {
  step->post_action_is_high_stakes = false;
  if (step->is_learner) {
    step->post_action_is_high_stakes =
        dune_imperium::IsHighStakesStrategicState(*ctx->state, step->current_player);
  }

  if (ctx->dune_state != nullptr) {
    for (int p = 0; p < ctx->game->NumPlayers(); ++p) {
      if (!ctx->had_swordmaster[p] && ctx->dune_state->HasSwordmaster(p)) {
        ctx->had_swordmaster[p] = true;
        if (ctx->local_stats != nullptr) {
          int purchase_round = ctx->dune_state->GetCurrentRound();
          int leader = ctx->dune_state->PlayerLeader(p);
          ctx->local_stats->sm_acquisitions_by_seat[p]++;
          if (leader >= 0 && leader < 14) {
            ctx->local_stats->sm_acquisitions_by_leader[leader]++;
          }
          if (purchase_round >= 1 && purchase_round <= 10) {
            ctx->local_stats->sm_acquisitions_by_round[purchase_round]++;
          }
        }
      }
    }
  }
}

void ApplyStepTrackingAndShaping(PpoSimContext* ctx, const SimStepDecision& step) {
  if (ctx->dune_state == nullptr) return;

  const Player current_player = step.current_player;
  const Action action = step.action;

  if (step.is_learner) {
    if (current_player >= 0 && current_player < ctx->game->NumPlayers() &&
        action != dune_imperium::kActionCombatPass) {
      int post_strength = dune_imperium::CombatStrength(*ctx->dune_state, current_player);
      int strength_delta = post_strength - ctx->pre_combat_strength[current_player];
      if (strength_delta > 0) {
        ctx->combat_accumulator.RecordDeployment(
            current_player, ctx->trajectory->size() - 1, strength_delta);
      }
    }

    if (ctx->dune_state->IsPlayerInRevealTurn(current_player) && ctx->local_stats != nullptr) {
      if (action == dune_imperium::kActionEndTurn) {
        ctx->local_stats->r755_reveal_turns++;
        ctx->local_stats->r755_unspent_persuasion_sum +=
            ctx->dune_state->PlayerPersuasion(current_player);
      } else if ((action >= dune_imperium::kActionBuyImperiumRow0 &&
                  action <= dune_imperium::kActionBuyImperiumRow0 + 7) ||
                 (action >= dune_imperium::kActionBuyHelenaReserve0 &&
                  action <= dune_imperium::kActionBuyHelenaReserve0 + 7) ||
                 action == dune_imperium::kActionBuyReserveArrakisLiaison ||
                 action == dune_imperium::kActionBuyReserveTheSpiceMustFlow) {
        ctx->local_stats->r755_imperium_buys++;
      } else if (action >= dune_imperium::kActionTleilaxuAcquire0 &&
                 action < dune_imperium::kActionTleilaxuAcquireSkip) {
        ctx->local_stats->r755_tleilaxu_buys++;
      } else if ((action >= dune_imperium::kActionTechAcquire0 &&
                  action < dune_imperium::kActionTechAcquireSkip) ||
                 (action >= dune_imperium::kActionTechAcquireWithSolari0 &&
                  action < dune_imperium::kActionTechAcquireWithSolariSkip)) {
        ctx->local_stats->r755_tech_buys++;
      }
    }

    // Specimen-exchange anti-breadcrumb (ConvertSpecimenToTroop, IDs 741-752):
    // uses shared unit-tested helpers; a positive penalty decreases reward.
    if (dune_shaping::IsSpecimenConversionAction(action)) {
      if (ctx->local_stats != nullptr) ctx->local_stats->specimen_conversions++;
      if (ctx->specimen_exchange_penalty != 0.0 && current_player >= 0 &&
          current_player < ctx->game->NumPlayers()) {
        int idx = ctx->last_transition_index[current_player];
        if (idx >= 0 && idx < static_cast<int>(ctx->trajectory->size())) {
          float before_rew = (*ctx->trajectory)[idx].reward;
          (*ctx->trajectory)[idx].reward = ApplySpecimenExchangeShaping(
              before_rew, action, ctx->specimen_exchange_penalty, ctx->reward_lambda);
          if (ctx->local_stats != nullptr) {
            ctx->local_stats->specimen_penalty_deducted +=
                (before_rew - (*ctx->trajectory)[idx].reward);
          }
        }
      }
    }

    if (step.is_atomics_action) {
      if (step.atomics_exempt) {
        if (ctx->local_stats != nullptr) ctx->local_stats->atomics_exempt++;
      } else {
        if (ctx->local_stats != nullptr) ctx->local_stats->atomics_penalized++;
        if (ctx->family_atomics_penalty != 0.0 && current_player >= 0 &&
            current_player < ctx->game->NumPlayers()) {
          int idx = ctx->last_transition_index[current_player];
          if (idx >= 0 && idx < static_cast<int>(ctx->trajectory->size())) {
            float penalty_amount =
                static_cast<float>(ctx->family_atomics_penalty) * ctx->reward_lambda;
            (*ctx->trajectory)[idx].reward -= penalty_amount;
            if (ctx->local_stats != nullptr) {
              ctx->local_stats->atomics_penalty_deducted += penalty_amount;
            }
          }
        }
      }
    }

    if (step.is_plot_intrigue_action) {
      if (step.plot_intrigue_exempt) {
        if (ctx->local_stats != nullptr) ctx->local_stats->plot_exempt++;
      } else {
        if (ctx->local_stats != nullptr) ctx->local_stats->plot_penalized++;
        if (ctx->plot_intrigue_penalty != 0.0 && current_player >= 0 &&
            current_player < ctx->game->NumPlayers()) {
          int idx = ctx->last_transition_index[current_player];
          if (idx >= 0 && idx < static_cast<int>(ctx->trajectory->size())) {
            float penalty_amount =
                static_cast<float>(ctx->plot_intrigue_penalty) * ctx->reward_lambda;
            (*ctx->trajectory)[idx].reward -= penalty_amount;
            if (ctx->local_stats != nullptr) {
              ctx->local_stats->plot_penalty_deducted += penalty_amount;
            }
          }
        }
      }
    }
  }

  for (int p = 0; p < ctx->game->NumPlayers(); ++p) {
    if (!ctx->seat_assignment.is_learner_seat[p]) continue;
    int new_tleilaxu = ctx->dune_state->GetTleilaxuTrackForTesting(p);
    int tleilaxu_delta = new_tleilaxu - ctx->current_tleilaxu[p];
    ctx->current_tleilaxu[p] = new_tleilaxu;
    if (tleilaxu_delta > 0 &&
        (ctx->tleilaxu_breadcrumb_weight > 0.0 ||
         ctx->tleilaxu_level7_breadcrumb_weight > 0.0)) {
      float tleilaxu_reward = 0.0f;
      int start_l = new_tleilaxu - tleilaxu_delta;
      for (int l = start_l + 1; l <= new_tleilaxu; ++l) {
        if (l >= 1 && l <= 6) {
          tleilaxu_reward +=
              static_cast<float>(ctx->tleilaxu_breadcrumb_weight) * ctx->reward_lambda;
        } else if (l == 7) {
          tleilaxu_reward +=
              static_cast<float>(ctx->tleilaxu_level7_breadcrumb_weight) * ctx->reward_lambda;
        }
      }
      int idx = ctx->last_transition_index[p];
      if (idx >= 0 && idx < static_cast<int>(ctx->trajectory->size())) {
        (*ctx->trajectory)[idx].reward += tleilaxu_reward;
      }
    }
  }

  for (int p = 0; p < ctx->game->NumPlayers(); ++p) {
    if (!ctx->seat_assignment.is_learner_seat[p]) continue;
    int raw_conflict_vp_delta =
        ctx->dune_state->ConflictVpDelta(p) - ctx->prev_conflict_vp[p];
    ctx->prev_conflict_vp[p] = ctx->dune_state->ConflictVpDelta(p);

    int raw_total_vp_delta = ctx->dune_state->GetPlayerVp(p) - ctx->prev_total_vp[p];
    ctx->prev_total_vp[p] = ctx->dune_state->GetPlayerVp(p);

    int raw_noncombat = raw_total_vp_delta - raw_conflict_vp_delta;

    if (ctx->local_stats != nullptr) {
      ctx->local_stats->raw_conflict_vp += raw_conflict_vp_delta;
      ctx->local_stats->raw_noncombat_vp += raw_noncombat;
      ctx->local_stats->raw_total_vp += raw_total_vp_delta;
    }

    float combat_shape = std::max(raw_conflict_vp_delta, 0) *
                         static_cast<float>(ctx->shaped_weight) * ctx->reward_lambda;
    float noncombat_shape = raw_noncombat *
                            static_cast<float>(ctx->shaped_weight) * ctx->reward_lambda;

    if (noncombat_shape != 0.0f) {
      int idx = ctx->last_transition_index[p];
      if (idx >= 0 && idx < static_cast<int>(ctx->trajectory->size())) {
        (*ctx->trajectory)[idx].reward += noncombat_shape;
      }
    }

    if (combat_shape > 0.0f) {
      if (ctx->local_stats != nullptr) {
        ctx->local_stats->conflict_vp_generated += combat_shape;
      }
      int total_investment = ctx->combat_accumulator.GetTotalInvestment(p);
      if (total_investment > 0) {
        for (const auto& ev : ctx->combat_accumulator.GetEvents(p)) {
          if (ev.transition_index >= 0 &&
              ev.transition_index < static_cast<int>(ctx->trajectory->size())) {
            float attributed_portion = combat_shape *
                                       static_cast<float>(ev.strength_delta) /
                                       static_cast<float>(total_investment);
            (*ctx->trajectory)[ev.transition_index].reward += attributed_portion;
            if (ctx->local_stats != nullptr) {
              ctx->local_stats->conflict_vp_attributed += attributed_portion;
            }
          }
        }
      } else {
        if (ctx->local_stats != nullptr) {
          ctx->local_stats->conflict_vp_unattributed += combat_shape;
        }
      }
    }
  }

  if (ctx->pre_action_phase == dune_imperium::GamePhase::kCombat &&
      ctx->dune_state->phase() != dune_imperium::GamePhase::kCombat) {
    ctx->combat_accumulator.ClearAll();
  }
}

namespace {

int FindStepActionIndex(const std::vector<Action>& actions, Action target) {
  for (size_t i = 0; i < actions.size(); ++i) {
    if (actions[i] == target) return static_cast<int>(i);
  }
  return -1;
}

void RecordR765OverrideProbe(PpoSimContext* ctx, const SimStepDecision& step,
                             uint64_t root_uid, const char* category,
                             Action a_star, Action a_ref, double delta_q,
                             double realized_kl, double realized_temp,
                             double base_mpo_temp,
                             const std::vector<double>& norm_v) {
  if (absl::GetFlag(FLAGS_r765_absorption_diag_jsonl).empty() ||
      ctx->local_stats == nullptr) {
    return;
  }
  R765OverrideProbeRecord probe;
  probe.root_uid = root_uid;
  probe.category = category;
  probe.observation = step.obs;
  probe.legal_actions = step.actions;
  if (step.p_cand_data != nullptr) probe.candidate_data = *step.p_cand_data;
  probe.a_star = a_star;
  probe.a_ref = a_ref;
  probe.delta_q = delta_q;
  probe.realized_kl = realized_kl;
  probe.realized_temp = realized_temp;
  const double base_t = std::max(1e-4, base_mpo_temp);
  probe.kl_cap_bound = (realized_temp > base_t + 1e-9);

  int s_idx = FindStepActionIndex(step.actions, a_star);
  int r_idx = FindStepActionIndex(step.actions, a_ref);
  if (s_idx >= 0 && r_idx >= 0 &&
      s_idx < static_cast<int>(step.legal_probabilities.size()) &&
      r_idx < static_cast<int>(step.legal_probabilities.size())) {
    double p_star = step.legal_probabilities[s_idx];
    double p_ref = step.legal_probabilities[r_idx];
    probe.t0_log_odds =
        std::log(std::max(1e-12, p_star)) - std::log(std::max(1e-12, p_ref));
  }
  if (s_idx >= 0 && r_idx >= 0 &&
      s_idx < static_cast<int>(norm_v.size()) &&
      r_idx < static_cast<int>(norm_v.size())) {
    double t_star = norm_v[s_idx];
    double t_ref = norm_v[r_idx];
    probe.target_log_odds =
        std::log(std::max(1e-12, t_star)) - std::log(std::max(1e-12, t_ref));
    size_t max_t_idx = std::max_element(norm_v.begin(), norm_v.end()) - norm_v.begin();
    probe.target_flips_argmax = (max_t_idx == static_cast<size_t>(s_idx));
  }
  ctx->local_stats->r765_override_probes.push_back(std::move(probe));
}

}  // namespace

void MaybeRecordR765AgreedProbe(PpoSimContext* ctx, const SimStepDecision& step,
                                Action greedy_action) {
  if (absl::GetFlag(FLAGS_r765_absorption_diag_jsonl).empty() ||
      ctx->local_stats == nullptr ||
      ctx->local_stats->r765_agreed_probes.size() >= 16) {
    return;
  }
  R765AgreedProbeRecord ap;
  ap.observation = step.obs;
  ap.legal_actions = step.actions;
  if (step.p_cand_data != nullptr) ap.candidate_data = *step.p_cand_data;
  ap.greedy_action = greedy_action;
  ctx->local_stats->r765_agreed_probes.push_back(std::move(ap));
}

void EmitR765PairwiseMpoAuxAndProbe(PpoSimContext* ctx, const SimStepDecision& step,
                                    uint64_t root_uid, const char* probe_category,
                                    const open_spiel::ActionsAndProbs& raw_policy_pairs,
                                    Action a_ref, double q_ref,
                                    Action a_star, double q_star) {
  if (ctx->r755_aux_examples == nullptr) return;
  double realized_kl = 0.0;
  double realized_temp = 0.0;
  const double mpo_temp = absl::GetFlag(FLAGS_r765_mpo_temperature);
  open_spiel::SearchTrainingExample ex;
  ex.observation = step.obs;
  ex.player = step.current_player;
  ex.legal_actions = step.actions;
  if (step.p_cand_data != nullptr) ex.candidate_data = *step.p_cand_data;
  ex.normalized_visits = open_spiel::dune_imperium::ComputePairwiseMpoTarget(
      raw_policy_pairs, step.actions, a_ref, q_ref, a_star, q_star,
      mpo_temp, absl::GetFlag(FLAGS_r765_mpo_max_kl), &realized_kl, &realized_temp);
  ex.best_action = a_star;
  ex.raw_action = a_ref;
  ex.value_target = 0.0;
  ex.value_target_attached = false;
  ex.episode_id = ctx->episode_id;
  ex.decision_id = static_cast<int>(ctx->r755_decision_index);
  ex.r765_root_uid = root_uid;
  ctx->r755_aux_examples->push_back(std::move(ex));

  RecordR765OverrideProbe(ctx, step, root_uid, probe_category, a_star, a_ref,
                          q_star - q_ref, realized_kl, realized_temp, mpo_temp,
                          ctx->r755_aux_examples->back().normalized_visits);
}

void EmitR755CompoundAuxAndProbe(
    PpoSimContext* ctx, const SimStepDecision& step, uint64_t root_uid,
    open_spiel::dune_imperium::R755SearchCategory cat,
    const open_spiel::ActionsAndProbs& raw_policy_pairs,
    const open_spiel::dune_imperium::CompoundSearchDecisionResult& search_res) {
  if (ctx->r755_aux_examples == nullptr) return;
  double realized_kl = 0.0;
  double realized_temp = 0.0;
  open_spiel::SearchTrainingExample ex;
  ex.observation = step.obs;
  ex.player = step.current_player;
  ex.legal_actions = step.actions;
  if (step.p_cand_data != nullptr) ex.candidate_data = *step.p_cand_data;
  const bool use_cat3_mpo =
      (cat == open_spiel::dune_imperium::R755SearchCategory::kRevealPurchase &&
       !search_res.stage2_candidate_utilities.empty());
  const double base_t = use_cat3_mpo ? absl::GetFlag(FLAGS_r755_cat3_mpo_temperature)
                                     : absl::GetFlag(FLAGS_r765_mpo_temperature);
  if (use_cat3_mpo) {
    ex.normalized_visits = open_spiel::dune_imperium::ComputeMpoRelativeTarget(
        raw_policy_pairs, step.actions, search_res.stage2_candidate_utilities,
        base_t, absl::GetFlag(FLAGS_r755_cat3_mpo_max_kl), &realized_kl, &realized_temp);
  } else {
    ex.normalized_visits = open_spiel::dune_imperium::ComputePairwiseMpoTarget(
        raw_policy_pairs, step.actions, search_res.raw_action,
        search_res.raw_mean_utility, search_res.best_action,
        search_res.chosen_mean_utility, base_t,
        absl::GetFlag(FLAGS_r765_mpo_max_kl), &realized_kl, &realized_temp);
  }
  if (search_res.overridden && search_res.chosen_action != search_res.raw_action) {
    ex.best_action = search_res.chosen_action;
    ex.raw_action = search_res.raw_action;
  }
  ex.value_target = 0.0;
  ex.value_target_attached = false;
  ex.episode_id = ctx->episode_id;
  ex.decision_id = static_cast<int>(ctx->r755_decision_index);
  ex.r765_root_uid = root_uid;
  ctx->r755_aux_examples->push_back(std::move(ex));

  if (search_res.overridden && search_res.chosen_action != search_res.raw_action) {
    const char* probe_cat =
        (cat == open_spiel::dune_imperium::R755SearchCategory::kRevealPurchase)
            ? "cat3_purchase"
            : "same_card_space";
    RecordR765OverrideProbe(
        ctx, step, root_uid, probe_cat, search_res.chosen_action,
        search_res.raw_action,
        search_res.chosen_mean_utility - search_res.raw_mean_utility,
        realized_kl, realized_temp, base_t,
        ctx->r755_aux_examples->back().normalized_visits);
  }
}

void DumpR755RootJsonl(
    uint64_t episode_id, int64_t decision_index, int cat_idx,
    const open_spiel::dune_imperium::CompoundSearchDecisionResult& search_res,
    double max_raw_p, double p_2) {
  const std::string root_dump_path = absl::GetFlag(FLAGS_r755_root_dump_jsonl);
  if (root_dump_path.empty()) return;
  static std::mutex dump_mutex;
  std::lock_guard<std::mutex> lock(dump_mutex);
  std::ofstream out(root_dump_path, std::ios::app);
  if (!out.is_open()) return;
  out << "{\"episode_id\":" << episode_id
      << ",\"decision_id\":" << decision_index
      << ",\"cat\":" << cat_idx
      << ",\"raw_action\":" << search_res.raw_action
      << ",\"best_action\":" << search_res.best_action
      << ",\"chosen_action\":" << search_res.chosen_action
      << ",\"overridden\":" << (search_res.overridden ? 1 : 0)
      << ",\"stopped_stage1\":" << (search_res.stopped_stage1 ? 1 : 0)
      << ",\"resample_fallbacks\":" << search_res.resample_fallbacks
      << ",\"nn_evals\":" << search_res.rollout_nn_evals
      << ",\"max_raw_p\":" << max_raw_p
      << ",\"p_2\":" << p_2
      << ",\"cand_actions\":[";
  for (size_t ci = 0; ci < search_res.candidate_rollout_utilities.size(); ++ci) {
    if (ci > 0) out << ",";
    out << search_res.candidate_rollout_utilities[ci].first;
  }
  out << "],\"cand_priors\":[";
  for (size_t ci = 0; ci < search_res.candidate_priors.size(); ++ci) {
    if (ci > 0) out << ",";
    out << search_res.candidate_priors[ci].second;
  }
  out << "],\"cand_utils\":[";
  for (size_t ci = 0; ci < search_res.candidate_rollout_utilities.size(); ++ci) {
    if (ci > 0) out << ",";
    out << "[";
    const auto& utils = search_res.candidate_rollout_utilities[ci].second;
    for (size_t ui = 0; ui < utils.size(); ++ui) {
      if (ui > 0) out << ",";
      out << utils[ui];
    }
    out << "]";
  }
  out << "]}\n";
}

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH

}  // namespace open_spiel::ppo_train_internal
