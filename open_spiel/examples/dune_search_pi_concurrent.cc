#include "dune_search_pi_concurrent.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel_utils.h"
#include "dune_batched_evaluator.h"
#include "dune_evaluator.h"
#include "dune_sha256.h"

namespace open_spiel {
namespace {

void MergeRole(SearchPiRoleStats* dst, const SearchPiRoleStats& src) {
  dst->roots_seen += src.roots_seen;
  dst->searches_run += src.searches_run;
  dst->rows_emitted += src.rows_emitted;
  dst->fallbacks += src.fallbacks;
  dst->simulations_completed += src.simulations_completed;
  dst->simulations_requested += src.simulations_requested;
  dst->searches_short_of_budget += src.searches_short_of_budget;
  dst->simulation_shortfall += src.simulation_shortfall;
  for (int i = 0; i < kSearchPiEarlyExitCount; ++i) {
    dst->early_exit_counts[i] += src.early_exit_counts[i];
  }
  if (dst->first_short_episode_id < 0 && src.first_short_episode_id >= 0) {
    dst->first_short_episode_id = src.first_short_episode_id;
    dst->first_short_decision_id = src.first_short_decision_id;
    dst->first_short_simulations_completed =
        src.first_short_simulations_completed;
    dst->first_short_simulations_requested =
        src.first_short_simulations_requested;
    dst->first_short_bot_reason = src.first_short_bot_reason;
    dst->first_short_session_reason = src.first_short_session_reason;
  }
  MergeSearchPiLatencyStats(dst, src);
  dst->inherited_visits += src.inherited_visits;
  dst->re_root_hits += src.re_root_hits;
  dst->re_root_misses += src.re_root_misses;
  dst->kept_despite_legacy_gate += src.kept_despite_legacy_gate;
  dst->sum_target_entropy_norm += src.sum_target_entropy_norm;
  dst->sum_raw_entropy_norm += src.sum_raw_entropy_norm;
  dst->sum_kl_target_given_raw += src.sum_kl_target_given_raw;
  dst->target_argmax_overrides += src.target_argmax_overrides;
}

const SearchPiRoleStats* RoleStats(
    const ConcurrentSearchPiCollectionResult& result, int role) {
  switch (role) {
    case 2: return &result.primary;
    case 3: return &result.continuation;
    case 4: return &result.purchase;
    case 5: return &result.combat_intrigue;
    case 6: return &result.other_optional;
    default: return nullptr;
  }
}

std::string ShapeString(const torch::Tensor& tensor) {
  std::ostringstream out;
  out << "[";
  for (int64_t i = 0; i < tensor.dim(); ++i) {
    if (i) out << ",";
    out << tensor.size(i);
  }
  out << "]";
  return out.str();
}

}  // namespace

bool SearchPiBatcherRowsIdentityOk(const BatcherTelemetry& telemetry) {
  return telemetry.submitted_rows ==
         telemetry.single_row_calls + telemetry.group_rows;
}

bool SearchPiBatcherDeviceTimingComplete(const BatcherTelemetry& telemetry,
                                         bool device_applicable) {
  return !device_applicable ||
         telemetry.device_timed_batches == telemetry.physical_batches;
}

ConcurrentSearchPiCollectionResult CollectSearchPiConcurrent(
    const ConcurrentSearchPiCollectionConfig& config,
    const std::shared_ptr<const Game>& game,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& frozen_model,
    torch::Device device) {
  SPIEL_CHECK_TRUE(game != nullptr);
  SPIEL_CHECK_TRUE(frozen_model != nullptr);
  SPIEL_CHECK_EQ(game->NumPlayers(), kSearchPiBatchedLeafGroupRows);
  SPIEL_CHECK_GT(config.collection_games, 0);
  SPIEL_CHECK_EQ(config.collection_games % 4, 0);
  SPIEL_CHECK_GT(config.chunk_games, 0);
  SPIEL_CHECK_EQ(config.chunk_games % 4, 0);
  SPIEL_CHECK_EQ(config.collection_games % config.chunk_games, 0);
  SPIEL_CHECK_GT(config.requested_workers, 0);
  const int num_chunks = config.collection_games / config.chunk_games;
  SPIEL_CHECK_LE(config.requested_workers, num_chunks);
  SPIEL_CHECK_EQ(config.warmup_games % 4, 0);
  SPIEL_CHECK_TRUE(!config.output_dir.empty());
  if (config.batch_target != 0) {
    SPIEL_CHECK_GT(config.batch_target, 0);
    SPIEL_CHECK_EQ(config.batch_target % 4, 0);
    SPIEL_CHECK_LE(config.batch_target,
                   kSearchPiBatchedLeafGroupRows * config.requested_workers);
    SPIEL_CHECK_GT(config.batcher_timeout_ms, 0);
  }

  std::error_code dir_error;
  std::filesystem::create_directories(config.output_dir, dir_error);
  SPIEL_CHECK_TRUE(std::filesystem::is_directory(config.output_dir));
  const std::string short_path =
      config.output_dir + "/" + SearchPiArmName(config.arm) +
      "_short_episodes.jsonl";
  std::ofstream short_stream(short_path, std::ios::trunc);
  if (!short_stream) {
    SpielFatalError(absl::StrCat("cannot open short-episode stream: ",
                                short_path));
  }

  const bool batched = config.batch_target > 0;
  std::shared_mutex model_mutex;

  // Warm-up is isolated in a throwaway evaluator so scored telemetry begins at
  // zero. The registered post-Gate-8 lane fixes warmup_games=0, but retaining
  // this exact audited behavior keeps the helper a drop-in audit consumer.
  if (config.warmup_games > 0) {
    torch::InferenceMode inference_guard;
    std::shared_ptr<BatchedEvaluator> warm_batched;
    std::shared_ptr<algorithms::Evaluator> warm_evaluator;
    if (batched) {
      warm_batched = std::make_shared<BatchedEvaluator>(
          frozen_model, config.batch_target, config.batcher_timeout_ms, device,
          &model_mutex, /*logit_cap=*/0.0f);
      warm_evaluator = std::make_shared<BatchedNNEvaluator>(
          warm_batched, config.logit_cap);
    } else {
      warm_evaluator = std::make_shared<DuneNNEvaluator>(
          frozen_model, device, config.logit_cap);
    }
    SearchPiConfig warm_config = config.search;
    warm_config.games_per_generation = config.warmup_games;
    warm_config.next_episode_id = 900000;
    std::vector<SearchPiRow> warm_rows;
    SearchPiGenerationStats warm_stats;
    SearchPiGenerator(warm_config).GenerateGeneration(
        config.generation, game, warm_evaluator, &warm_rows, &warm_stats,
        config.arm);
    warm_evaluator.reset();
    warm_batched.reset();
  }

  const auto wall_start = std::chrono::steady_clock::now();
  std::shared_ptr<BatchedEvaluator> shared_batcher;
  if (batched) {
    shared_batcher = std::make_shared<BatchedEvaluator>(
        frozen_model, config.batch_target, config.batcher_timeout_ms, device,
        &model_mutex, /*logit_cap=*/0.0f);
    shared_batcher->EnableBatcherTelemetry();
  }

  std::vector<std::shared_ptr<algorithms::Evaluator>> evaluators;
  evaluators.reserve(config.requested_workers);
  for (int worker = 0; worker < config.requested_workers; ++worker) {
    if (batched) {
      evaluators.push_back(std::make_shared<BatchedNNEvaluator>(
          shared_batcher, config.logit_cap));
    } else {
      evaluators.push_back(std::make_shared<DuneNNEvaluator>(
          frozen_model, device, config.logit_cap));
    }
  }

  std::vector<SearchPiGenerationStats> chunk_stats(num_chunks);
  std::vector<std::vector<SearchPiRow>> chunk_rows(
      config.retain_rows ? num_chunks : 0);
  std::atomic<int> next_chunk{0};
  std::mutex output_mutex;
  auto worker_fn = [&](int slot) {
    torch::InferenceMode inference_guard;
    const std::shared_ptr<algorithms::Evaluator> evaluator = evaluators[slot];
    for (;;) {
      const int chunk_index = next_chunk.fetch_add(1);
      if (chunk_index >= num_chunks) break;
      SearchPiConfig chunk_config = config.search;
      chunk_config.games_per_generation = config.chunk_games;
      chunk_config.next_episode_id =
          config.search.next_episode_id +
          static_cast<int64_t>(chunk_index) * config.chunk_games;
      std::vector<SearchPiRow> rows;
      SearchPiGenerator(chunk_config).GenerateGeneration(
          config.generation, game, evaluator, &rows,
          &chunk_stats[chunk_index], config.arm);
      if (config.retain_rows) chunk_rows[chunk_index] = std::move(rows);
      {
        std::lock_guard<std::mutex> lock(output_mutex);
        for (const SearchPiGameOutcome& outcome :
             chunk_stats[chunk_index].games_played) {
          if (outcome.searches_short_of_budget > 0) {
            short_stream << "{\"episode_id\":" << outcome.episode_id
                         << ",\"root_fires\":"
                         << outcome.searches_short_of_budget << "}\n";
          }
        }
        short_stream.flush();
        if (!short_stream) {
          SpielFatalError(absl::StrCat(
              "write failed for short-episode stream: ", short_path));
        }
        std::cout << "[search-PI concurrent] " << SearchPiArmName(config.arm)
                  << " chunk " << (chunk_index + 1) << "/" << num_chunks
                  << " episodes [" << chunk_stats[chunk_index].first_episode_id
                  << ".." << chunk_stats[chunk_index].next_episode_id << ")"
                  << " rows=" << chunk_stats[chunk_index].rows_total << "\n";
      }
    }
  };

  std::vector<std::thread> workers;
  for (int slot = 1; slot < config.requested_workers; ++slot) {
    workers.emplace_back(worker_fn, slot);
  }
  worker_fn(0);
  for (std::thread& worker : workers) worker.join();

  ConcurrentSearchPiCollectionResult result;
  result.arm = config.arm;
  result.requested_workers = config.requested_workers;
  result.actual_workers = config.requested_workers;
  result.configured_batch_target = config.batch_target;
  result.configured_batcher_timeout_ms = config.batcher_timeout_ms;
  result.max_inflight_rows =
      batched ? kSearchPiBatchedLeafGroupRows * config.requested_workers : 0;

  // Merge only in chunk/episode order. Completion order is scientifically
  // irrelevant and must not alter rows, hashes, or first-fire attribution.
  for (int chunk_index = 0; chunk_index < num_chunks; ++chunk_index) {
    const SearchPiGenerationStats& stats = chunk_stats[chunk_index];
    MergeRole(&result.primary, stats.primary);
    MergeRole(&result.continuation, stats.continuation);
    MergeRole(&result.purchase, stats.purchase);
    MergeRole(&result.combat_intrigue, stats.combat_intrigue);
    MergeRole(&result.other_optional, stats.other_optional);
    for (int role = 0; role < 7; ++role) {
      result.simulations_by_role[role] += stats.simulations_by_role[role];
      result.decisions_by_role[role] += stats.decisions_by_role[role];
    }
    result.leader_rows_emitted += stats.leader_rows_emitted;
    result.inference_calls += stats.inference_calls;
    result.rows_total += stats.rows_total;
    result.games.insert(result.games.end(), stats.games_played.begin(),
                        stats.games_played.end());
  }
  if (config.retain_rows) {
    for (int chunk_index = 0; chunk_index < num_chunks; ++chunk_index) {
      for (const SearchPiRow& row : chunk_rows[chunk_index]) {
        result.target_hash_chain =
            ChainSearchPiTargetHash(result.target_hash_chain, row);
        result.extended_hash_chain =
            ChainSearchPiExtendedRowHash(result.extended_hash_chain, row);
        result.retained_rows.push_back(row);
      }
    }
  }
  if (batched) {
    result.telemetry = shared_batcher->GetBatcherTelemetry();
    result.telemetry_valid = true;
    result.telemetry_device_applicable = device.is_cuda();
  }
  result.wall_time_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - wall_start).count();
  return result;
}

SearchPiTrainingCollectionValidation ValidateSearchPiTrainingCollection(
    const ConcurrentSearchPiCollectionResult& result,
    const SearchPiTrainingCollectionContract& contract) {
  SearchPiTrainingCollectionValidation validation;
  auto fail = [&](const std::string& error) {
    validation.errors.push_back(error);
  };
  const int64_t collection_end =
      contract.first_episode_id + contract.collection_games;
  const int64_t training_end =
      contract.first_episode_id + contract.training_games;
  const std::vector<int> searched_roles =
      contract.purchase_combat_budget > 0
          ? std::vector<int>{2, 3, 4, 5, 6}
          : std::vector<int>{2, 3};

  // Selection happens first and consults only immutable episode IDs.
  for (const SearchPiRow& row : result.retained_rows) {
    if (row.episode_id < contract.first_episode_id ||
        row.episode_id >= collection_end) {
      fail(absl::StrCat("row outside registered collection range: ",
                        row.episode_id));
      continue;
    }
    validation.collected_target_hash = ChainSearchPiTargetHash(
        validation.collected_target_hash, row);
    validation.collected_extended_hash = ChainSearchPiExtendedRowHash(
        validation.collected_extended_hash, row);
    const int collected_role = static_cast<int>(row.role);
    if (std::find(searched_roles.begin(), searched_roles.end(),
                  collected_role) == searched_roles.end()) {
      fail(absl::StrCat("collected row has out-of-scope role ",
                        collected_role));
    }
    if (row.episode_id < training_end) {
      validation.training_rows.push_back(row);
      validation.trained_target_hash = ChainSearchPiTargetHash(
          validation.trained_target_hash, row);
      validation.trained_extended_hash = ChainSearchPiExtendedRowHash(
          validation.trained_extended_hash, row);
      const int role = static_cast<int>(row.role);
      if (role < 0 || role >= 7) {
        fail(absl::StrCat("trained row has invalid role ", role));
      } else {
        validation.trained_rows_by_role[role]++;
      }
      if (row.simulations_completed != row.simulations_requested) {
        fail(absl::StrCat("training row episode ", row.episode_id,
                          " is short: ", row.simulations_completed, "/",
                          row.simulations_requested));
      }
      if (row.early_exit != SearchPiEarlyExit::kNone ||
          row.fallback != SearchPiFallback::kNone) {
        fail(absl::StrCat("training row episode ", row.episode_id,
                          " has an early exit or fallback"));
      }
    }
  }

  if (contract.collection_games != 512 || contract.training_games != 64) {
    fail("collection/training geometry is not the registered 512/64");
  }
  if (contract.training_games % 4 != 0) {
    fail("training prefix is not four-game seat balanced");
  }
  if (result.requested_workers != contract.requested_workers ||
      result.actual_workers != contract.requested_workers) {
    fail(absl::StrCat("worker contract mismatch requested/actual ",
                      result.requested_workers, "/", result.actual_workers,
                      " expected ", contract.requested_workers));
  }
  if (result.configured_batch_target != contract.batch_target ||
      result.configured_batcher_timeout_ms != contract.batcher_timeout_ms ||
      result.max_inflight_rows !=
          kSearchPiBatchedLeafGroupRows * contract.requested_workers) {
    fail("batch target/timeout/capacity contract mismatch");
  }
  if (!result.telemetry_valid ||
      !SearchPiBatcherRowsIdentityOk(result.telemetry) ||
      !SearchPiBatcherDeviceTimingComplete(
          result.telemetry, result.telemetry_device_applicable) ||
      result.telemetry.leaf_groups_split != 0 ||
      result.telemetry.target_batch_size !=
          static_cast<uint64_t>(contract.batch_target) ||
      result.telemetry.timeout_ms !=
          static_cast<uint64_t>(contract.batcher_timeout_ms) ||
      result.telemetry.max_batch_size >
          static_cast<uint64_t>(contract.batch_target)) {
    fail("batcher telemetry/invariant failure");
  }
  if (result.games.size() != static_cast<size_t>(contract.collection_games)) {
    fail(absl::StrCat("collected ", result.games.size(), " games, expected ",
                      contract.collection_games));
  }
  if (result.rows_total != static_cast<int64_t>(result.retained_rows.size())) {
    fail("rows_total differs from retained row count");
  }
  if (result.leader_rows_emitted != 0 ||
      result.simulations_by_role[0] != 0 ||
      result.simulations_by_role[1] != 0) {
    fail("forced/bookkeeping role was searched or emitted");
  }

  int64_t played_differences_by_role[7] = {0, 0, 0, 0, 0, 0, 0};
  int64_t game_primary_roots = 0, game_primary_done = 0,
          game_primary_requested = 0;
  int64_t game_continuation_roots = 0, game_continuation_done = 0,
          game_continuation_requested = 0;
  int64_t game_wide_roots = 0, game_wide_done = 0,
          game_wide_requested = 0;
  int64_t game_fallbacks = 0, game_timeouts = 0, game_shorts = 0;

  int64_t previous_episode = contract.first_episode_id - 1;
  for (const SearchPiGameOutcome& game : result.games) {
    if (game.episode_id != previous_episode + 1) {
      fail(absl::StrCat("games are missing, duplicated, or out of order at ",
                        game.episode_id));
    }
    previous_episode = game.episode_id;
    for (int role = 0; role < 7; ++role) {
      played_differences_by_role[role] +=
          game.roots_played_differs_from_raw[role];
    }
    game_primary_roots += game.primary_roots;
    game_primary_done += game.primary_simulations;
    game_primary_requested += game.primary_simulations_requested;
    game_continuation_roots += game.continuation_roots;
    game_continuation_done += game.continuation_simulations;
    game_continuation_requested += game.continuation_simulations_requested;
    game_wide_roots += game.wide_roots;
    game_wide_done += game.wide_simulations;
    game_wide_requested += game.wide_simulations_requested;
    game_fallbacks += game.fallbacks;
    game_timeouts += game.watchdog_timeouts;
    game_shorts += game.searches_short_of_budget;
    const bool training = game.episode_id < training_end;
    if (game.off_scope_simulations != 0 ||
        game.non_timeout_early_exits != 0 ||
        game.fallbacks != game.watchdog_fallbacks ||
        game.watchdog_fallbacks > game.watchdog_timeouts ||
        game.searches_short_of_budget != game.watchdog_timeouts) {
      fail(absl::StrCat("episode ", game.episode_id,
                        " has a non-timeout early exit, malformed attribution, "
                        "or off-scope simulation"));
      continue;
    }
    const bool exact =
        game.primary_simulations == game.primary_simulations_requested &&
        game.primary_simulations_requested ==
            game.primary_roots * contract.primary_simulations &&
        game.continuation_simulations ==
            game.continuation_simulations_requested &&
        game.continuation_simulations_requested ==
            game.continuation_roots * contract.continuation_simulations &&
        game.wide_simulations == game.wide_simulations_requested &&
        game.wide_simulations_requested ==
            game.wide_roots * contract.purchase_combat_budget;
    if (training) {
      if (game.watchdog_timeouts != 0 || !exact) {
        fail(absl::StrCat("training-subset episode ", game.episode_id,
                          " fired the watchdog or missed an exact budget"));
      }
    } else if (game.watchdog_timeouts != 0) {
      validation.filler_timeout_episode_ids.push_back(game.episode_id);
    } else if (!exact) {
      fail(absl::StrCat("non-timeout filler episode ", game.episode_id,
                        " missed an exact budget"));
    }
  }
  if (previous_episode + 1 != collection_end) {
    fail("collected episode range does not end at the registered boundary");
  }
  if (game_primary_roots != result.primary.roots_seen ||
      game_primary_done != result.primary.simulations_completed ||
      game_primary_requested != result.primary.simulations_requested ||
      game_continuation_roots != result.continuation.roots_seen ||
      game_continuation_done != result.continuation.simulations_completed ||
      game_continuation_requested !=
          result.continuation.simulations_requested) {
    fail("per-episode primary/continuation accounting does not reconcile");
  }
  const int64_t wide_roots = result.purchase.roots_seen +
                             result.combat_intrigue.roots_seen +
                             result.other_optional.roots_seen;
  const int64_t wide_done = result.purchase.simulations_completed +
                            result.combat_intrigue.simulations_completed +
                            result.other_optional.simulations_completed;
  const int64_t wide_requested = result.purchase.simulations_requested +
                                 result.combat_intrigue.simulations_requested +
                                 result.other_optional.simulations_requested;
  if (game_wide_roots != wide_roots || game_wide_done != wide_done ||
      game_wide_requested != wide_requested) {
    fail("per-episode wide-role accounting does not reconcile");
  }
  const int64_t summary_fallbacks = result.primary.fallbacks +
      result.continuation.fallbacks + result.purchase.fallbacks +
      result.combat_intrigue.fallbacks + result.other_optional.fallbacks;
  const int timeout_index = static_cast<int>(SearchPiEarlyExit::kTimeout);
  const int64_t summary_timeouts = result.primary.early_exit_counts[timeout_index] +
      result.continuation.early_exit_counts[timeout_index] +
      result.purchase.early_exit_counts[timeout_index] +
      result.combat_intrigue.early_exit_counts[timeout_index] +
      result.other_optional.early_exit_counts[timeout_index];
  const int64_t summary_shorts = result.primary.searches_short_of_budget +
      result.continuation.searches_short_of_budget +
      result.purchase.searches_short_of_budget +
      result.combat_intrigue.searches_short_of_budget +
      result.other_optional.searches_short_of_budget;
  if (game_fallbacks != summary_fallbacks || game_timeouts != summary_timeouts ||
      game_shorts != summary_shorts) {
    fail("per-episode fallback/timeout/short counts do not reconcile");
  }
  if (validation.filler_timeout_episode_ids.size() >
      static_cast<size_t>(contract.max_filler_timeout_episodes)) {
    fail(absl::StrCat("filler watchdog discard cap exceeded: ",
                      validation.filler_timeout_episode_ids.size(), " > ",
                      contract.max_filler_timeout_episodes));
  }
  if (validation.training_rows.empty()) {
    fail("fixed training prefix emitted zero rows");
  }
  for (int role = 0; role < 7; ++role) {
    const bool searched = std::find(searched_roles.begin(), searched_roles.end(),
                                    role) != searched_roles.end();
    if (!searched && validation.trained_rows_by_role[role] != 0) {
      fail(absl::StrCat("out-of-scope role ", role,
                        " reached the learner subset"));
    }
    const SearchPiRoleStats* role_stats = RoleStats(result, role);
    if (!searched) {
      if (result.simulations_by_role[role] != 0) {
        fail(absl::StrCat("out-of-scope role ", role,
                          " ran simulations"));
      }
      continue;
    }
    if (role_stats == nullptr) {
      fail(absl::StrCat("searched role ", role, " has no statistics bucket"));
      continue;
    }
    const int dose = role == 2 ? contract.primary_simulations
                     : role == 3 ? contract.continuation_simulations
                                 : contract.purchase_combat_budget;
    if (result.decisions_by_role[role] != role_stats->roots_seen ||
        result.simulations_by_role[role] !=
            role_stats->simulations_completed ||
        role_stats->simulations_requested != role_stats->roots_seen * dose) {
      fail(absl::StrCat("role ", role,
                        " decision/root/completed/requested accounting mismatch"));
    }
    if (role_stats->roots_seen > 0) {
      if (role_stats->configured_time_limit_ms != 60000.0) {
        fail(absl::StrCat("role ", role,
                          " did not run under the 60000 ms watchdog"));
      }
      if (played_differences_by_role[role] <= 0) {
        fail(absl::StrCat("role ", role,
                          " searched but never changed the played action"));
      }
    }
  }
  validation.ok = validation.errors.empty();
  return validation;
}

std::string CanonicalSearchPiModuleDigest(torch::nn::Module& model) {
  struct Entry {
    std::string category;
    std::string name;
    torch::Tensor tensor;
  };
  std::vector<Entry> entries;
  for (const auto& item : model.named_parameters(/*recurse=*/true)) {
    entries.push_back({"parameter", item.key(), item.value()});
  }
  for (const auto& item : model.named_buffers(/*recurse=*/true)) {
    entries.push_back({"buffer", item.key(), item.value()});
  }
  std::sort(entries.begin(), entries.end(), [](const Entry& lhs,
                                                const Entry& rhs) {
    return std::tie(lhs.category, lhs.name) < std::tie(rhs.category, rhs.name);
  });
  std::string canonical = "search_pi_module_digest_v1\n";
  for (const Entry& entry : entries) {
    torch::Tensor tensor = entry.tensor.detach().to(torch::kCPU).contiguous();
    canonical += entry.category + "\t" + entry.name + "\t" +
                 ShapeString(tensor) + "\t" +
                 std::string(c10::toString(tensor.scalar_type())) + "\t" +
                 absl::StrCat(tensor.numel() * tensor.element_size()) + "\n";
    const size_t bytes =
        static_cast<size_t>(tensor.numel()) * tensor.element_size();
    canonical.append(static_cast<const char*>(tensor.data_ptr()), bytes);
    canonical.push_back('\n');
  }
  return ComputeStringSHA256(canonical);
}

}  // namespace open_spiel
