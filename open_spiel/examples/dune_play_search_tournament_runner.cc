#include "open_spiel/examples/dune_play_search_tournament_internal.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "open_spiel/examples/dune_seed_utils.h"

namespace open_spiel::dune_imperium::play_search_tournament {
namespace {

bool IsLeader(const State& state) {
  const auto& dune = static_cast<const DuneImperiumState&>(state);
  return dune.phase() == GamePhase::kLeaderDraft ||
         dune.phase() == GamePhase::kLeaderOfferChance;
}

int PriorRank(const ActionsAndProbs& prior, const std::vector<Action>& legal,
              Action chosen) {
  const Action raw = PickGreedyAction(prior, legal);
  if (chosen == raw) return 1;
  std::vector<std::pair<double, Action>> scored;
  for (Action a : legal) {
    if (a == raw) continue;
    double probability = 0;
    for (const auto& item : prior) if (item.first == a) probability = item.second;
    scored.emplace_back(probability, a);
  }
  std::sort(scored.rbegin(), scored.rend());
  for (size_t i = 0; i < scored.size(); ++i) if (scored[i].second == chosen) return i + 2;
  throw std::runtime_error("Controller selected an illegal action");
}

CompoundSearchDecisionResult SearchDecision(
    const State& state, Player seat, TurnSearchTree* tree,
    const std::vector<std::shared_ptr<BatchedNNEvaluator>>& evaluators,
    const Options& o, uint64_t seed, int effective_rollouts_override = -1,
    double max_time_ms = -1.0) {
  const int eff_rollouts =
      (effective_rollouts_override > 0) ? effective_rollouts_override : o.rollouts;
  if (o.mode == Mode::kShSuperhuman) {
    return SuccessiveHalvingRolloutSearchDecision(
        state, seat, tree, evaluators, eff_rollouts, kOverrideMargin, seed,
        o.crn_mode, o.unbiased_stage2_gate, max_time_ms,
        o.fixed_top5_candidates ? 5 : 0, /*max_inflight_worlds=*/128,
        /*use_bounded_pool_when_unclocked=*/false,
        /*full_clock_stage2=*/(max_time_ms > 0.0),
        o.prior_log_ratio_weight, o.stage1_high_conf_prior_skip,
        /*precomputed_raw_prior=*/nullptr, o.all_buy_candidates,
        o.win_speed_tiebreak_eps, o.ceiling_aware_margin);
  }
  if (o.mode == Mode::kShAdaptive || o.mode == Mode::kShAdaptiveLeader ||
      o.mode == Mode::kFullShAdaptiveLeader) {
    return SuccessiveHalvingRolloutSearchDecision(
        state, seat, tree, evaluators, eff_rollouts, kOverrideMargin, seed,
        o.crn_mode, o.unbiased_stage2_gate, max_time_ms,
        o.fixed_top5_candidates ? 5 : 0, /*max_inflight_worlds=*/96,
        /*use_bounded_pool_when_unclocked=*/false, /*full_clock_stage2=*/false,
        o.prior_log_ratio_weight, o.stage1_high_conf_prior_skip,
        /*precomputed_raw_prior=*/nullptr, o.all_buy_candidates,
        o.win_speed_tiebreak_eps, o.ceiling_aware_margin);
  }
  if (o.mode == Mode::kAdaptive || o.mode == Mode::kAdaptiveLeader) {
    return AdaptiveRolloutSearchDecision(
        state, seat, tree, evaluators, eff_rollouts, kOverrideMargin, seed,
        o.crn_mode, max_time_ms, /*max_inflight_worlds=*/32,
        /*precomputed_raw_prior=*/nullptr, o.win_speed_tiebreak_eps,
        o.ceiling_aware_margin);
  }
  if (o.mode == Mode::kWide) {
    return WideRolloutSearchDecision(
        state, seat, tree, evaluators, eff_rollouts, kOverrideMargin, seed,
        o.crn_mode, o.win_speed_tiebreak_eps, o.ceiling_aware_margin);
  }
  CompoundSearchOptions opts;
  opts.all_buy_candidates = o.all_buy_candidates;
  opts.win_speed_tiebreak_eps = o.win_speed_tiebreak_eps;
  opts.ceiling_aware_margin = o.ceiling_aware_margin;
  return CompoundRolloutSearchDecision(
      state, seat, tree, evaluators, 3, eff_rollouts, kOverrideMargin,
      TreeReuseMode::kAdditive, seed, SearchPairingMode::kUnpaired,
      SearchTruncationMode::kTerminal, o.crn_mode, nullptr, 0.0, opts);
}

bool ShouldSearchSuperhumanState(const State& state, Player seat,
                                 const ActionsAndProbs& prior,
                                 const std::vector<Action>& legal,
                                 Action raw_action,
                                 const TurnSearchTree& tree) {
  if (legal.size() <= 1 || state.CurrentPlayer() != seat) return false;
  if (IsHighStakesStrategicState(state, seat) || IsPurchaseDecisionState(state)) {
    return true;
  }
  const auto& dune = static_cast<const DuneImperiumState&>(state);
  if (dune.phase() == GamePhase::kCombat) {
    return true;
  }
  bool has_end_or_reveal = false;
  for (Action a : legal) {
    if (a == kActionEndTurn || a == kActionReveal) {
      has_end_or_reveal = true;
      break;
    }
  }
  if (!has_end_or_reveal && !tree.HasConfirmedSubtree(raw_action, 16)) {
    return true;
  }
  if (has_end_or_reveal && IsFullStrategicState(state, seat, prior, 0.020)) {
    return true;
  }
  return false;
}

std::vector<RootCase> ReadRoots(const Options& o) {
  const auto value = ReadJson(o.roots_path);
  if (!value.IsObject()) throw std::runtime_error("Root corpus must be an object");
  const auto& object = value.GetObject();
  auto roots = object.find("roots");
  if (roots == object.end() || !roots->second.IsArray()) throw std::runtime_error("Missing roots array");
  std::map<std::pair<int, std::string>, std::vector<RootCase>> strata;
  std::set<int> ids;
  for (const auto& item : roots->second.GetArray()) {
    if (!item.IsObject()) throw std::runtime_error("Root must be an object");
    const auto& row = item.GetObject();
    RootCase root{IntField(row, "root_id"), IntField(row, "game_id"),
                  IntField(row, "search_seat"), IntField(row, "round"),
                  StringField(row, "decision_type"), {}};
    if (!ids.insert(root.root_id).second || root.seat < 0 || root.seat >= 4) {
      throw std::runtime_error("Duplicate root ID or invalid seat");
    }
    auto history = row.find("action_history");
    if (history == row.end() || !history->second.IsArray()) throw std::runtime_error("Missing root history");
    for (const auto& action : history->second.GetArray()) {
      if (!action.IsInt()) throw std::runtime_error("Invalid action in root history");
      root.history.push_back(action.GetInt());
    }
    const int band = root.round <= 3 ? 0 : root.round <= 6 ? 1 : 2;
    strata[{band, root.decision_type}].push_back(std::move(root));
  }
  for (auto& [key, bucket] : strata) {
    std::sort(bucket.begin(), bucket.end(), [](const auto& a, const auto& b) { return a.root_id < b.root_id; });
  }
  std::vector<RootCase> result;
  for (size_t offset = 0;; ++offset) {
    bool added = false;
    for (const auto& [key, bucket] : strata) {
      if (offset < bucket.size()) {
        result.push_back(bucket[offset]);
        added = true;
        if (o.max_roots > 0 && result.size() == static_cast<size_t>(o.max_roots)) return result;
      }
    }
    if (!added) return result;
  }
}

}  // namespace

std::optional<GameOutcome> PlayGame(
    int game_id, const Game& game, const ModelBundle& candidate,
    const ModelBundle& opponent, const Options& o, const RunControl& control) {
  GameOutcome out;
  out.game_id = game_id;
  out.seat = game_id % 4;
  out.seed = dune_seed::DeriveSeed(o.master_seed, dune_seed::kStreamSearchSampling, game_id);
  const auto start = Clock::now();
  auto state = game.NewInitialState();
  std::mt19937_64 chance_rng(out.seed);
  TurnSearchTree tree;
  Player last_player = kInvalidPlayer;
  uint64_t search_seed_counter = 0;
  const std::vector<std::shared_ptr<BatchedNNEvaluator>> simulations{candidate.evaluator};
  std::array<int, 4> r7_track_vps = {0, 0, 0, 0};
  bool captured_r7_vp = false;
  double ms_per_rollout_ema = 12.0;

  while (!state->IsTerminal()) {
    if (control.Stop()) return std::nullopt;
    if (!captured_r7_vp) {
      const auto& cur_dune = static_cast<const DuneImperiumState&>(*state);
      if (cur_dune.GetCurrentRound() > 7) {
        for (int p = 0; p < 4; ++p) r7_track_vps[p] = cur_dune.GetPlayerVp(p);
        captured_r7_vp = true;
      }
    }
    if (state->IsChanceNode()) {
      const auto& before = static_cast<const DuneImperiumState&>(*state);
      const int hand = before.GetPlayerCardsInHand(out.seat);
      const int intrigues = before.GetPlayerIntrigues(out.seat).size();
      Action action = SampleAction(state->ChanceOutcomes(),
          std::generate_canonical<double, 53>(chance_rng)).first;
      state->ApplyAction(action);
      const auto& after = static_cast<const DuneImperiumState&>(*state);
      if (after.GetPlayerCardsInHand(out.seat) > hand ||
          static_cast<int>(after.GetPlayerIntrigues(out.seat).size()) > intrigues) {
        tree.Reset();
        ++out.new_info_resets;
      }
      continue;
    }
    const Player current = state->CurrentPlayer();
    if (current != last_player) {
      tree.Reset();
      last_player = current;
    }
    const auto legal = state->LegalActions();
    if (legal.empty()) throw std::runtime_error("Nonterminal state has no legal actions");
    auto& evaluator = current == out.seat ? candidate.evaluator : opponent.evaluator;
    const auto prior = evaluator->Prior(*state);
    if (prior.empty()) throw std::runtime_error("Empty neural prior");
    const Action raw = PickGreedyAction(prior, legal);
    Action chosen = raw;
    if (current == out.seat) {
      ++out.decisions;
      if ((o.mode == Mode::kLeader || o.mode == Mode::kAdaptiveLeader ||
           o.mode == Mode::kShAdaptiveLeader ||
           o.mode == Mode::kFullShAdaptiveLeader ||
           o.mode == Mode::kShSuperhuman) &&
          IsLeader(*state) && legal.size() > 1) {
        const auto search_start = Clock::now();
        int rollouts = o.leader_rollouts;
        double leader_deadline_ms = -1.0;
        if (o.mode == Mode::kShSuperhuman && o.game_clock_ms > 0.0) {
          const double rem_ms = std::max(0.0, o.game_clock_ms - 1000.0 * out.search_seconds);
          leader_deadline_ms = std::min(12000.0, 0.18 * rem_ms);
          rollouts = std::clamp(
              static_cast<int>((0.90 * leader_deadline_ms) /
                               std::max(2.0, ms_per_rollout_ema * static_cast<double>(legal.size()))),
              4, 512);
        }
        const uint64_t leader_seed =
            dune_seed::DeriveSeed(out.seed, 0x4C454144ULL, current);
        const auto result = LeaderPickRolloutSearchDecision(
            *state, current, simulations, rollouts, 0.10, leader_seed,
            o.crn_mode, o.sample_opponent_leader_draft, leader_deadline_ms,
            /*max_inflight_worlds=*/128, &prior, o.win_speed_tiebreak_eps,
            o.ceiling_aware_margin);
        const double step_s = SecondsSince(search_start);
        out.search_seconds += step_s;
        if (result.new_rollouts_executed >= 6 && step_s > 0.0) {
          ms_per_rollout_ema =
              0.8 * ms_per_rollout_ema +
              0.2 * ((1000.0 * step_s) / result.new_rollouts_executed);
        }
        if (control.Stop()) return std::nullopt;
        chosen = result.action;
        ++out.searches;
        out.overrides += result.overridden;
        out.new_rollouts += result.new_rollouts_executed;
        if (result.candidate_rollout_utilities.size() >= 2) {
          auto corr_stats = ComputeMeanPairedCorrelation(result.candidate_rollout_utilities);
          out.sum_paired_corr += corr_stats.sum_corr;
          out.sum_paired_diff_var += corr_stats.sum_diff_var;
          out.paired_corr_pairs += corr_stats.corr_pair_count;
          out.paired_diff_var_pairs += corr_stats.diff_var_pair_count;
        }
      } else {
        const bool search_allowed =
            o.mode != Mode::kRaw && o.mode != Mode::kLeader && !IsLeader(*state);
        if (o.mode == Mode::kShSuperhuman) {
          const bool should_search =
              search_allowed &&
              ShouldSearchSuperhumanState(*state, current, prior, legal, raw, tree);
          if (should_search) {
            int eff_rollouts = o.rollouts;
            double target_ms = -1.0;
            if (o.game_clock_ms > 0.0) {
              const auto& d_cur = static_cast<const DuneImperiumState&>(*state);
              const int cur_round = std::clamp(d_cur.GetCurrentRound(), 1, 10);
              const int rem_rounds = std::max(1, 8 - cur_round + 1);
              const double rem_ms =
                  std::max(0.0, 0.92 * o.game_clock_ms - 1000.0 * out.search_seconds);
              target_ms = std::min(15000.0, rem_ms / static_cast<double>(2 * rem_rounds));
              eff_rollouts = std::clamp(
                  static_cast<int>((0.85 * target_ms) / (3.0 * std::max(1.0, ms_per_rollout_ema))),
                  4, 1024);
            }
            if (o.game_clock_ms <= 0.0 || target_ms >= 25.0) {
              const auto search_start = Clock::now();
              const uint64_t seed = dune_seed::DeriveSeed(out.seed, current, search_seed_counter++);
              const auto result =
                  SearchDecision(*state, current, &tree, simulations, o, seed, eff_rollouts, target_ms);
              const double step_s = SecondsSince(search_start);
              out.search_seconds += step_s;
              if (result.new_rollouts_executed >= 6 && step_s > 0.0) {
                ms_per_rollout_ema =
                    0.8 * ms_per_rollout_ema +
                    0.2 * ((1000.0 * step_s) / result.new_rollouts_executed);
              }
              if (control.Stop()) return std::nullopt;
              chosen = result.action;
              ++out.searches;
              out.overrides += result.overridden;
              out.new_rollouts += result.new_rollouts_executed;
              out.reused_rollouts += result.prior_rollouts_reused;
              out.prior_rank4plus_selections += result.overridden && PriorRank(prior, legal, chosen) >= 4;
              if (result.candidate_rollout_utilities.size() >= 2) {
                auto corr_stats = ComputeMeanPairedCorrelation(result.candidate_rollout_utilities);
                out.sum_paired_corr += corr_stats.sum_corr;
                out.sum_paired_diff_var += corr_stats.sum_diff_var;
                out.paired_corr_pairs += corr_stats.corr_pair_count;
                out.paired_diff_var_pairs += corr_stats.diff_var_pair_count;
              }
            } else if (tree.current_node() && !tree.current_node()->children.empty()) {
              chosen = tree.BestConfirmedAction(raw, prior, legal);
            }
          } else if (search_allowed && tree.current_node() && !tree.current_node()->children.empty()) {
            chosen = tree.BestConfirmedAction(raw, prior, legal);
          }
        } else if (o.mode == Mode::kFullShAdaptiveLeader) {
          const bool should_search =
              search_allowed && IsFullStrategicState(*state, current, prior, 0.020);
          if (should_search) {
            const auto search_start = Clock::now();
            const uint64_t seed = dune_seed::DeriveSeed(out.seed, current, search_seed_counter++);
            const auto result = SearchDecision(*state, current, &tree, simulations, o, seed);
            out.search_seconds += SecondsSince(search_start);
            if (control.Stop()) return std::nullopt;
            chosen = result.action;
            ++out.searches;
            out.overrides += result.overridden;
            out.new_rollouts += result.new_rollouts_executed;
            out.reused_rollouts += result.prior_rollouts_reused;
            out.prior_rank4plus_selections += result.overridden && PriorRank(prior, legal, chosen) >= 4;
            if (result.candidate_rollout_utilities.size() >= 2) {
              auto corr_stats = ComputeMeanPairedCorrelation(result.candidate_rollout_utilities);
              out.sum_paired_corr += corr_stats.sum_corr;
              out.sum_paired_diff_var += corr_stats.sum_diff_var;
              out.paired_corr_pairs += corr_stats.corr_pair_count;
              out.paired_diff_var_pairs += corr_stats.diff_var_pair_count;
            }
          } else if (search_allowed && tree.current_node() && !tree.current_node()->children.empty()) {
            chosen = tree.BestConfirmedAction(raw, prior, legal);
          }
        } else if (search_allowed && IsHighStakesStrategicState(*state, current) && legal.size() > 1) {
          const auto search_start = Clock::now();
          const uint64_t seed = dune_seed::DeriveSeed(out.seed, current, search_seed_counter++);
          const auto result = SearchDecision(*state, current, &tree, simulations, o, seed);
          out.search_seconds += SecondsSince(search_start);
          if (control.Stop()) return std::nullopt;
          chosen = result.action;
          ++out.searches;
          out.overrides += result.overridden;
          out.new_rollouts += result.new_rollouts_executed;
          out.reused_rollouts += result.prior_rollouts_reused;
          if (o.mode == Mode::kWide && legal.size() > 3) {
            ++out.wide_roots;
          }
          out.prior_rank4plus_selections += result.overridden && PriorRank(prior, legal, chosen) >= 4;
          if (result.candidate_rollout_utilities.size() >= 2) {
            auto corr_stats = ComputeMeanPairedCorrelation(result.candidate_rollout_utilities);
            out.sum_paired_corr += corr_stats.sum_corr;
            out.sum_paired_diff_var += corr_stats.sum_diff_var;
            out.paired_corr_pairs += corr_stats.corr_pair_count;
            out.paired_diff_var_pairs += corr_stats.diff_var_pair_count;
          }
        } else if (search_allowed && tree.current_node() && !tree.current_node()->children.empty()) {
          chosen = tree.BestConfirmedAction(raw, prior, legal);
        }
      }
      if (o.mode != Mode::kRaw) {
        tree.Advance(chosen);
        if (chosen == kActionEndTurn) tree.Reset();
      }
    }
    if (std::find(legal.begin(), legal.end(), chosen) == legal.end()) {
      throw std::runtime_error("Selected action is not legal");
    }
    const auto& before = static_cast<const DuneImperiumState&>(*state);
    const int hand = current == out.seat ? before.GetPlayerCardsInHand(out.seat) : 0;
    const int intrigues = current == out.seat ? before.GetPlayerIntrigues(out.seat).size() : 0;
    state->ApplyAction(chosen);
    const auto& after = static_cast<const DuneImperiumState&>(*state);
    if (current == out.seat && chosen != kActionEndTurn &&
        (after.GetPlayerCardsInHand(out.seat) > hand ||
         static_cast<int>(after.GetPlayerIntrigues(out.seat).size()) > intrigues)) {
      tree.Reset();
      ++out.new_info_resets;
    }
  }
  out.returns = state->Returns();
  if (out.returns.size() != 4 ||
      !std::all_of(out.returns.begin(), out.returns.end(), [](double r) { return std::isfinite(r); })) {
    throw std::runtime_error("Invalid terminal returns");
  }
  out.placement = 1;
  for (double r : out.returns) out.placement += r > out.returns[out.seat];
  const auto& dune = static_cast<const DuneImperiumState&>(*state);
  for (int p = 0; p < 4; ++p) out.vps.push_back(dune.GetPlayerVp(p));
  out.ending_round = dune.GetCurrentRound();
  if (!captured_r7_vp) {
    for (int p = 0; p < 4; ++p) r7_track_vps[p] = dune.GetPlayerVp(p);
  }
  out.cand_r7_track_vp = r7_track_vps[out.seat];
  out.max4_r7_track_vp = *std::max_element(r7_track_vps.begin(), r7_track_vps.end());
  out.game_seconds = SecondsSince(start);
  return out;
}

json::Object BaseRecord(const Options& o, const std::string& kind) {
  json::Object rec{{"record_type", kind}, {"config_key", o.config_key}, {"mode", o.mode_name},
          {"controller_version", kControllerVersion}, {"simulation_policy", kSimulationPolicy},
          {"model_checkpoint", o.model_path}, {"opponent_checkpoint", o.opponent_path},
          {"master_seed", std::to_string(o.master_seed)}, {"rollouts_per_action", o.rollouts},
          {"margin", kOverrideMargin}, {"rollout_amp", o.rollout_amp},
          {"allow_tf32", o.allow_tf32}, {"device", o.device_name},
          {"threads", o.threads}, {"target_batch_size", o.batch_size},
          {"crn_mode", CrnModeName(o.crn_mode)},
          {"unbiased_stage2_gate", o.unbiased_stage2_gate},
          {"fixed_top5_candidates", o.fixed_top5_candidates},
          {"all_buy_candidates", o.all_buy_candidates},
          {"win_speed_tiebreak_eps", o.win_speed_tiebreak_eps},
          {"ceiling_aware_margin", o.ceiling_aware_margin},
          {"sample_opponent_leader_draft", o.sample_opponent_leader_draft},
          {"leader_rollouts", o.leader_rollouts}};
  if (o.kl_policy_tau > 0.0) rec["kl_policy_tau"] = o.kl_policy_tau;
  if (o.game_clock_ms > 0.0) rec["game_clock_ms"] = o.game_clock_ms;
  return rec;
}

json::Object GameRecord(const GameOutcome& out, const Options& o) {
  auto row = BaseRecord(o, "game");
  row["game_id"] = out.game_id;
  row["seed"] = std::to_string(out.seed);
  row["seat"] = out.seat;
  json::Array returns, vps;
  for (double r : out.returns) returns.emplace_back(r);
  for (int vp : out.vps) vps.emplace_back(vp);
  row["returns"] = returns;
  row["final_vps"] = vps;
  row["utility"] = out.returns[out.seat];
  row["placement"] = out.placement;
  row["ending_round"] = out.ending_round;
  row["cand_r7_track_vp"] = out.cand_r7_track_vp;
  row["max4_r7_track_vp"] = out.max4_r7_track_vp;
  row["frac_rounds_le_7"] = (out.ending_round <= 7) ? 1.0 : 0.0;
  row["searches"] = out.searches;
  row["overrides"] = out.overrides;
  row["new_rollouts"] = out.new_rollouts;
  row["reused_rollouts"] = out.reused_rollouts;
  row["new_info_resets"] = out.new_info_resets;
  row["decisions"] = out.decisions;
  row["wide_roots"] = out.wide_roots;
  row["prior_rank4plus_selections"] = out.prior_rank4plus_selections;
  row["search_wall_seconds"] = out.search_seconds;
  row["game_wall_seconds"] = out.game_seconds;
  const double mean_paired_corr = out.paired_corr_pairs > 0
                                      ? (out.sum_paired_corr / out.paired_corr_pairs)
                                      : 0.0;
  const double mean_paired_diff_var = out.paired_diff_var_pairs > 0
                                          ? (out.sum_paired_diff_var / out.paired_diff_var_pairs)
                                          : 0.0;
  row["sum_paired_corr"] = out.sum_paired_corr;
  row["sum_paired_diff_var"] = out.sum_paired_diff_var;
  row["paired_corr_pairs"] = out.paired_corr_pairs;
  row["paired_diff_var_pairs"] = out.paired_diff_var_pairs;
  row["mean_paired_rollout_corr"] = mean_paired_corr;
  row["mean_paired_diff_var"] = mean_paired_diff_var;
  return row;
}

int RunRoots(const Options& o, const Game& game, const ModelBundle& candidate,
             const RunControl& control, const std::set<int>& completed,
             std::ofstream& output) {
  const auto roots = ReadRoots(o);
  const std::vector<std::shared_ptr<BatchedNNEvaluator>> evaluators{candidate.evaluator};
  int done = 0;
  for (const auto& root : roots) {
    if (completed.count(root.root_id)) { ++done; continue; }
    if (control.Stop()) break;
    auto state = game.NewInitialState();
    for (size_t index = 0; index < root.history.size(); ++index) {
      const Action action = root.history[index];
      const auto legal = state->LegalActions();
      if (std::find(legal.begin(), legal.end(), action) == legal.end()) {
        throw std::runtime_error("Illegal root history: root_id=" + std::to_string(root.root_id) +
            " index=" + std::to_string(index) + " action=" + std::to_string(action) +
            " current_player=" + std::to_string(state->CurrentPlayer()));
      }
      state->ApplyAction(action);
    }
    if (state->IsTerminal() || state->IsChanceNode() || state->CurrentPlayer() != root.seat) {
      throw std::runtime_error("Replayed root has the wrong actor or is not a player decision");
    }
    const auto start = Clock::now();
    const auto nn_start = g_total_network_evaluations.load();
    const auto legal = state->LegalActions();
    const auto prior = candidate.evaluator->Prior(*state);
    const Action raw = PickGreedyAction(prior, legal);
    const uint64_t seed = dune_seed::DeriveSeed(o.master_seed, dune_seed::kStreamSearchSampling, root.root_id);
    CompoundSearchDecisionResult result;
    result.action = raw;
    TurnSearchTree tree;
    const bool should_search =
        o.mode != Mode::kRaw && o.mode != Mode::kLeader && !IsLeader(*state) &&
        (o.mode == Mode::kShSuperhuman
             ? ShouldSearchSuperhumanState(*state, root.seat, prior, legal, raw, tree)
             : (o.mode == Mode::kFullShAdaptiveLeader
                    ? IsFullStrategicState(*state, root.seat, prior, 0.020)
                    : (IsHighStakesStrategicState(*state, root.seat) && legal.size() > 1)));
    if (should_search) {
      result = SearchDecision(*state, root.seat, &tree, evaluators, o, seed);
    }
    if (control.Stop()) break;
    auto row = BaseRecord(o, "root");
    row["roots_json"] = o.roots_path;
    row["root_id"] = root.root_id;
    row["game_id"] = root.game_id;
    row["seed"] = std::to_string(seed);
    row["seat"] = root.seat;
    row["round"] = root.round;
    row["decision_type"] = root.decision_type;
    row["chosen_action"] = result.action;
    row["raw_action"] = raw;
    row["selected_prior_rank"] = PriorRank(prior, legal, result.action);
    const bool wide_applicable = should_search && o.mode == Mode::kWide && legal.size() > 3;
    row["wide_applicable"] = wide_applicable;
    row["overridden"] = result.overridden;
    row["new_rollouts"] = result.new_rollouts_executed;
    row["reused_rollouts"] = result.prior_rollouts_reused;
    row["search_wall_seconds"] = SecondsSince(start);
    row["network_evaluations"] = static_cast<int64_t>(g_total_network_evaluations.load() - nn_start);
    json::Array utilities;
    for (const auto& [action, utility] : result.candidate_utilities) {
      utilities.emplace_back(json::Array{action, utility});
    }
    row["candidate_utilities"] = utilities;
    double root_mean_paired_corr = 0.0;
    double root_mean_paired_diff_var = 0.0;
    if (result.candidate_rollout_utilities.size() >= 2) {
      auto corr_stats = ComputeMeanPairedCorrelation(result.candidate_rollout_utilities);
      root_mean_paired_corr = corr_stats.mean_paired_corr;
      root_mean_paired_diff_var = corr_stats.mean_paired_diff_var;
    }
    row["mean_paired_rollout_corr"] = root_mean_paired_corr;
    row["mean_paired_diff_var"] = root_mean_paired_diff_var;
    output << json::ToString(row) << '\n' << std::flush;
    if (!output) throw std::runtime_error("Failed to append root output");
    std::cout << "[ROOT] " << ++done << '/' << roots.size() << " id=" << root.root_id
              << " elapsed=" << SecondsSince(control.start) << "s\n" << std::flush;
  }
  return done == static_cast<int>(roots.size()) ? 0 : 2;
}

TournamentRunStats RunTournamentGames(
    const Options& o, const Game& game, const ModelBundle& candidate,
    const ModelBundle& opponent, RunControl& control,
    const std::vector<int>& pending, int skipped, Summary& summary,
    std::ofstream& output) {
  std::atomic<size_t> next{0};
  std::mutex writer;
  TournamentRunStats stats;
  const auto play_start = Clock::now();
  auto worker = [&]() {
    try {
      while (!control.Stop()) {
        const size_t index = next.fetch_add(1);
        if (index >= pending.size()) break;
        auto result = PlayGame(pending[index], game, candidate, opponent, o, control);
        if (!result) break;
        const auto row = GameRecord(*result, o);
        std::lock_guard<std::mutex> lock(writer);
        summary.completed_games++;
        if (result->placement == 1) summary.wins++;
        summary.sum_utility += result->returns[result->seat];
        summary.sum_ending_round += result->ending_round;
        if (result->ending_round <= 7) summary.rounds_le_7_count++;
        summary.sum_cand_r7_track_vp += result->cand_r7_track_vp;
        summary.sum_max4_r7_track_vp += result->max4_r7_track_vp;
        summary.sum_paired_corr += result->sum_paired_corr;
        summary.sum_paired_diff_var += result->sum_paired_diff_var;
        summary.paired_corr_pairs += result->paired_corr_pairs;
        summary.paired_diff_var_pairs += result->paired_diff_var_pairs;
        output << json::ToString(row) << '\n' << std::flush;
        if (!output) throw std::runtime_error("Failed to append completed game");
        const int done = ++stats.newly_completed;
        const double elapsed = SecondsSince(play_start);
        const double eta = elapsed / done * (pending.size() - done);
        std::cout << "[GAME] " << done + skipped << '/' << o.games
                  << " id=" << result->game_id << " seat=" << result->seat
                  << " placement=" << result->placement << " utility="
                  << result->returns[result->seat]
                  << " ending_round=" << result->ending_round
                  << " cand_r7_track_vp=" << result->cand_r7_track_vp
                  << " max4_r7_track_vp=" << result->max4_r7_track_vp
                  << " elapsed=" << elapsed
                  << "s eta=" << eta << "s\n" << std::flush;
      }
    } catch (const std::exception& e) {
      control.failed.store(true);
      std::lock_guard<std::mutex> lock(writer);
      if (stats.error.empty()) stats.error = e.what();
    }
  };
  std::vector<std::thread> workers;
  for (int i = 0; i < o.threads; ++i) workers.emplace_back(worker);
  for (auto& thread : workers) thread.join();
  return stats;
}

}  // namespace open_spiel::dune_imperium::play_search_tournament
