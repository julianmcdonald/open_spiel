// Frozen offline matched-dose changed-state weighting experiment.
//
// There is no game, bot, collector, supervisor, resume, or optimizer-load path
// in this executable.  It parses the ten retained F cohorts, constructs two
// fresh AdamW instances from byte-identical model loads, and changes only the
// row weights between the two one-epoch learners.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/spiel_utils.h"

#include <torch/torch.h>

#include "dune_offline_changed_state.h"
#include "dune_search_pi_replay.h"
#include "dune_sha256.h"

ABSL_FLAG(std::string, mode, "", "preflight or run");
ABSL_FLAG(std::string, shards, "", "Ten comma-separated retained F shards");
ABSL_FLAG(std::string, expected_shard_sizes, "",
          "Ten comma-separated registered byte sizes");
ABSL_FLAG(std::string, expected_shard_sha256, "",
          "Ten comma-separated registered SHA-256 values");
ABSL_FLAG(std::string, start_model, "", "Registered boot model checkpoint");
ABSL_FLAG(std::string, expected_start_sha256, "",
          "Registered boot model file SHA-256");
ABSL_FLAG(std::string, expected_start_digest, "",
          "Frozen canonical in-memory start digest (required in run mode)");
ABSL_FLAG(std::string, expected_target_hash, "",
          "Frozen combined target hash (required in run mode)");
ABSL_FLAG(std::string, expected_extended_hash, "",
          "Frozen combined extended-row hash (required in run mode)");
ABSL_FLAG(std::string, expected_binary_sha256, "",
          "Frozen executable SHA-256 (required in run mode)");
ABSL_FLAG(std::string, optimizer_checkpoint, "",
          "Always refused; proves there is no optimizer restore boundary");
ABSL_FLAG(std::string, device, "auto", "auto, cuda, or cpu");
ABSL_FLAG(std::string, output, "", "Preflight/result JSON path");
ABSL_FLAG(std::string, output_dir, "", "Runtime model output directory");

// Link-only definitions required by dune_ppo_training_utils.cc.  The offline
// lane calls only CenterAndCapLogitsTensor; none of these flags is read.
ABSL_FLAG(int, ppo_minibatch_size, 2048, "INERT here (link satisfaction)");
ABSL_FLAG(int, ppo_update_epochs, 4, "INERT here (link satisfaction)");
ABSL_FLAG(double, ppo_clip_epsilon, 0.2, "INERT here (link satisfaction)");
ABSL_FLAG(bool, normalize_advantages, true, "INERT here (link satisfaction)");
ABSL_FLAG(bool, ppo_clip_value_loss, true, "INERT here (link satisfaction)");
ABSL_FLAG(double, entropy_coef, 0.01, "INERT here (link satisfaction)");
ABSL_FLAG(double, value_coef, 0.5, "INERT here (link satisfaction)");
ABSL_FLAG(double, target_kl, 0.0, "INERT here (link satisfaction)");
ABSL_FLAG(bool, train_amp, true, "INERT here (link satisfaction)");
ABSL_FLAG(double, grad_clip_norm, 0.5, "INERT here (link satisfaction)");
ABSL_FLAG(double, logit_cap, 10.0, "INERT here (link satisfaction)");
ABSL_FLAG(uint64_t, shaping_start_env_steps, 206830543,
          "INERT here (link satisfaction)");
ABSL_FLAG(uint64_t, shaping_decay_env_steps, 0,
          "INERT here (link satisfaction)");
ABSL_FLAG(bool, diagnostics_only, false, "INERT here (link satisfaction)");

namespace open_spiel {
namespace {

constexpr int64_t kObservationSize = 5580;
constexpr int64_t kActionDim = 2391;
constexpr int64_t kHiddenDim = 2048;
constexpr int kResidualBlocks = 8;
constexpr const char* kTerminationLabel =
    "PRINCIPAL-TERMINATED AFTER GEN10 — registered 4N endpoint not reached; "
    "no confirmatory 4N verdict.";

struct ResolvedShard {
  std::string path;
  uint64_t expected_size = 0;
  std::string expected_sha256;
  uint64_t actual_size = 0;
  std::string actual_sha256;
  int generation = 0;
  int64_t rows = 0;
  std::string target_hash;
  std::string extended_hash;
};

std::string JsonString(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    switch (c) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (c < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(c) << std::dec << std::setfill(' ');
        } else {
          out << static_cast<char>(c);
        }
    }
  }
  return out.str() + '"';
}

void WriteAtomically(const std::string& path, const std::string& content) {
  const std::filesystem::path target(path);
  if (!target.has_parent_path()) {
    SpielFatalError("offline output path must have a parent directory");
  }
  std::filesystem::create_directories(target.parent_path());
  const std::filesystem::path temp = target.string() + ".tmp";
  {
    std::ofstream out(temp, std::ios::out | std::ios::trunc);
    if (!out) SpielFatalError("cannot open offline temp output: " + temp.string());
    out << content << '\n';
    out.flush();
    if (!out) SpielFatalError("cannot write offline temp output: " + temp.string());
  }
  std::filesystem::rename(temp, target);
}

std::vector<std::string> Csv(const std::string& value) {
  return absl::StrSplit(value, ',');
}

std::vector<uint64_t> SizeCsv(const std::string& value) {
  std::vector<uint64_t> out;
  for (const std::string& item : Csv(value)) {
    if (item.empty()) SpielFatalError("empty expected shard-size field");
    size_t consumed = 0;
    const uint64_t parsed = std::stoull(item, &consumed);
    if (consumed != item.size()) {
      SpielFatalError("invalid expected shard size: " + item);
    }
    out.push_back(parsed);
  }
  return out;
}

torch::Device ResolveDevice(const std::string& requested, bool preflight) {
  if (preflight || requested == "cpu") return torch::Device(torch::kCPU);
  if (requested == "cuda") {
    if (!torch::cuda::is_available()) {
      SpielFatalError("--device=cuda requested but CUDA is unavailable");
    }
    return torch::Device(torch::kCUDA);
  }
  if (requested == "auto") {
    return torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                       : torch::Device(torch::kCPU);
  }
  SpielFatalError("--device must be auto, cuda, or cpu");
}

std::shared_ptr<SharedDunePolicyValueNetImpl> LoadModel(
    const std::string& path, torch::Device device) {
  auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
      kObservationSize, kHiddenDim, kActionDim, kResidualBlocks,
      /*use_nonlinear=*/false, /*with_aux_heads=*/false);
  torch::serialize::InputArchive archive;
  archive.load_from(path, device);
  model->load(archive);
  model->to(device);
  model->eval();
  return model;
}

void ValidateParsedRow(const SearchPiRow& row, int generation,
                       size_t stored_index, const std::string& shard_path) {
  const std::string where = shard_path + " stored row " +
                            std::to_string(stored_index);
  if (row.generation != generation) {
    SpielFatalError("cohort generation mismatch in " + where);
  }
  if (row.observation.empty() ||
      static_cast<int64_t>(row.observation.size()) > kObservationSize) {
    SpielFatalError("invalid observation width in " + where);
  }
  for (float value : row.observation) {
    if (!std::isfinite(value)) {
      SpielFatalError("non-finite observation value in " + where);
    }
  }
  if (row.legal_actions.empty() ||
      row.legal_actions.size() != row.target_probs.size() ||
      row.legal_actions.size() != row.raw_policy.size()) {
    SpielFatalError("unaligned legal actions/policies in " + where);
  }
  std::set<Action> legal;
  double target_mass = 0.0;
  int raw_argmax = -1;
  int target_argmax = -1;
  double raw_max = 0.0;
  double target_max = 0.0;
  for (size_t i = 0; i < row.legal_actions.size(); ++i) {
    const Action action = row.legal_actions[i];
    if (action < 0 || action >= kActionDim || !legal.insert(action).second) {
      SpielFatalError("invalid or duplicate legal action in " + where);
    }
    const double target = row.target_probs[i];
    const double raw = row.raw_policy[i];
    if (!std::isfinite(target) || target < 0.0 || !std::isfinite(raw) ||
        raw < 0.0) {
      SpielFatalError("invalid raw/target probability in " + where);
    }
    target_mass += target;
    if (raw_argmax < 0 || raw > raw_max) {
      raw_argmax = static_cast<int>(i);
      raw_max = raw;
    }
    if (target_argmax < 0 || target > target_max) {
      target_argmax = static_cast<int>(i);
      target_max = target;
    }
  }
  if (std::abs(target_mass - 1.0) > 1e-9) {
    SpielFatalError("target probability mass is not one in " + where);
  }
  if (row.target_argmax_differs_from_raw !=
      (raw_argmax != target_argmax)) {
    SpielFatalError("stored changed/preserved label disagrees with policies in " +
                    where);
  }
}

void SaveModelAtomically(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model,
    const std::string& path) {
  const std::filesystem::path target(path);
  std::filesystem::create_directories(target.parent_path());
  const std::filesystem::path temp = target.string() + ".tmp";
  std::error_code ignored;
  std::filesystem::remove(temp, ignored);
  torch::save(model, temp.string());
  std::filesystem::rename(temp, target);
}

std::string MetricLevelJson(const OfflineMetricLevel& level) {
  std::ostringstream out;
  out << std::setprecision(17)
      << "{\"n\":" << level.n << ",\"ce_mean\":" << level.ce_mean
      << ",\"teacher_argmax_adopted_count\":" << level.adopted
      << ",\"teacher_argmax_adoption\":" << level.adoption << '}';
  return out.str();
}

std::string MeanIntervalJson(const OfflineMeanInterval& interval) {
  std::ostringstream out;
  out << std::setprecision(17)
      << "{\"n\":" << interval.n << ",\"mean\":" << interval.mean
      << ",\"sd\":" << interval.sd << ",\"se\":" << interval.se
      << ",\"lower\":" << interval.lower << ",\"upper\":"
      << interval.upper
      << ",\"method\":\"paired Wald 95% interval over row-level "
         "differences\"}";
  return out.str();
}

std::string PairedJson(const OfflinePairedComparison& paired) {
  std::ostringstream out;
  out << std::setprecision(17)
      << "{\"ce_t_minus_c\":" << MeanIntervalJson(paired.ce_t_minus_c)
      << ",\"adoption_t_minus_c\":"
      << MeanIntervalJson(paired.adoption_t_minus_c)
      << ",\"adoption_gained\":" << paired.adoption_gained
      << ",\"adoption_lost\":" << paired.adoption_lost
      << ",\"adoption_discordant\":" << paired.adoption_discordant
      << ",\"adoption_one_sided_exact_p\":"
      << paired.adoption_one_sided_exact_p
      << ",\"adoption_test\":\"one-sided exact paired binomial; "
         "alternative T adoption greater than C\"}";
  return out.str();
}

std::string OptimizerContractJson(const OfflineAdamWContract& contract) {
  std::ostringstream out;
  out << std::setprecision(17) << "{\"groups\":[";
  for (size_t i = 0; i < contract.groups.size(); ++i) {
    if (i) out << ',';
    const OfflineAdamWGroupContract& group = contract.groups[i];
    out << "{\"index\":" << i << ",\"parameter_names\":[";
    for (size_t j = 0; j < group.parameter_names.size(); ++j) {
      if (j) out << ',';
      out << JsonString(group.parameter_names[j]);
    }
    out << "],\"options\":{\"learning_rate\":" << group.learning_rate
        << ",\"betas\":[" << group.beta1 << ',' << group.beta2 << ']'
        << ",\"epsilon\":" << group.epsilon
        << ",\"weight_decay\":" << group.weight_decay
        << ",\"amsgrad\":" << (group.amsgrad ? "true" : "false")
        << "}}";
  }
  return out.str() + "]}";
}

std::string PreflightJson(
    const std::string& mode, const std::string& binary_path,
    const std::string& binary_sha256, const std::string& start_path,
    uint64_t start_size, const std::string& start_sha256,
    const std::string& start_digest,
    const std::vector<ResolvedShard>& shards, int64_t rows,
    const std::string& target_hash, const std::string& extended_hash,
    const OfflineWeightPlan& weights,
    const std::vector<OfflineBatchBoundary>& boundaries,
    const std::string& presentation_digest,
    const OfflineAdamWContract& optimizer_contract) {
  std::ostringstream out;
  out << std::setprecision(17)
      << "{\"schema_version\":1,\"mode\":" << JsonString(mode)
      << ",\"analysis_kind\":\"FROZEN_OFFLINE_CHANGED_STATE_WEIGHTING\""
      << ",\"confirmatory\":false,\"termination_label\":"
      << JsonString(kTerminationLabel)
      << ",\"binary\":{\"path\":" << JsonString(binary_path)
      << ",\"size\":" << std::filesystem::file_size(binary_path)
      << ",\"sha256\":" << JsonString(binary_sha256) << '}'
      << ",\"start_model\":{\"path\":" << JsonString(start_path)
      << ",\"size\":" << start_size << ",\"sha256\":"
      << JsonString(start_sha256) << ",\"canonical_digest\":"
      << JsonString(start_digest) << '}'
      << ",\"shards\":[";
  for (size_t i = 0; i < shards.size(); ++i) {
    if (i) out << ',';
    const ResolvedShard& shard = shards[i];
    out << "{\"generation\":" << shard.generation << ",\"path\":"
        << JsonString(shard.path) << ",\"size\":" << shard.actual_size
        << ",\"sha256\":" << JsonString(shard.actual_sha256)
        << ",\"parsed_rows\":" << shard.rows
        << ",\"target_hash\":" << JsonString(shard.target_hash)
        << ",\"extended_hash\":" << JsonString(shard.extended_hash)
        << '}';
  }
  out << "],\"corpus\":{\"rows\":" << rows
      << ",\"target_hash\":" << JsonString(target_hash)
      << ",\"extended_hash\":" << JsonString(extended_hash)
      << ",\"order\":\"cohort generation ascending, then stored row "
         "order\",\"roles\":[";
  for (size_t i = 0; i < weights.roles.size(); ++i) {
    if (i) out << ',';
    const OfflineRoleWeightSummary& role = weights.roles[i];
    out << "{\"role\":" << JsonString(role.spec.name)
        << ",\"total\":" << role.parsed_total
        << ",\"changed\":" << role.parsed_changed
        << ",\"preserved\":" << role.parsed_preserved
        << ",\"changed_weight\":" << role.changed_weight
        << ",\"preserved_weight\":" << role.preserved_weight
        << ",\"changed_loss_mass\":" << role.changed_loss_mass
        << ",\"preserved_loss_mass\":" << role.preserved_loss_mass
        << ",\"total_loss_mass\":" << role.total_loss_mass << '}';
  }
  out << "],\"control_loss_mass\":" << weights.control_mass
      << ",\"treatment_loss_mass\":" << weights.treatment_mass
      << ",\"treatment_mean_weight\":"
      << weights.treatment_mass / rows << '}'
      << ",\"presentation\":{\"seed\":"
      << kOfflineChangedStateSeed
      << ",\"algorithm\":\"fixed contiguous stored order; no sampling, "
         "duplication, or shuffle\",\"batch_size\":"
      << kOfflineChangedStateBatchSize
      << ",\"minibatches\":" << boundaries.size()
      << ",\"final_minibatch_size\":"
      << (boundaries.empty() ? 0 : boundaries.back().count)
      << ",\"digest\":" << JsonString(presentation_digest) << '}'
      << ",\"learner\":{\"epochs\":1,\"policy_ce_only\":true,"
         "\"value_coefficient\":0,\"learning_rate\":"
      << kOfflineChangedStateLearningRate << ",\"adam_epsilon\":"
      << kOfflineChangedStateAdamEps
      << ",\"adam_betas\":[0.9,0.999],\"policy_weight_decay\":"
      << kOfflineChangedStatePolicyWeightDecay
      << ",\"other_weight_decay\":" << kOfflineChangedStateWeightDecay
      << ",\"gradient_clip\":" << kOfflineChangedStateGradientClip
      << ",\"legal_centered_tanh_logit_cap\":"
      << kOfflineChangedStateLogitCap
      << ",\"weighted_ce_reduction\":\"sum(w_i * CE_i) / "
         "sum(w_i) per fixed minibatch\"}"
      << ",\"optimizer_contract\":"
      << OptimizerContractJson(optimizer_contract)
      << ",\"gate_methods\":{\"ce_intervals\":\"paired row-level Wald "
         "95% intervals\",\"adoption_noninferiority_interval\":\"paired "
         "row-level Wald 95% interval\",\"changed_adoption_test\":\"one-"
         "sided exact paired binomial\"}"
      << ",\"assertions\":{\"start_file_hash_match\":true,"
         "\"start_canonical_digests_equal\":true,"
         "\"fresh_adamw_state_maps_empty\":true,"
         "\"optimizer_group_membership_and_options_equal\":true,"
         "\"optimizer_checkpoint_load_refused\":true,"
         "\"shard_bytes_and_hashes_match\":true,"
         "\"row_target_and_extended_hashes_recorded\":true,"
         "\"parsed_counts_and_loss_masses_match\":true,"
         "\"arm_presentation_contracts_equal\":true}}";
  return out.str();
}

std::string TrainingStatsJson(const OfflineTrainingStats& stats) {
  std::ostringstream out;
  out << std::setprecision(17)
      << "{\"presentations\":" << stats.presentations
      << ",\"minibatches\":" << stats.minibatches
      << ",\"final_minibatch_size\":" << stats.final_minibatch_size
      << ",\"presentation_weight_mass\":"
      << stats.presentation_weight_mass
      << ",\"weighted_ce_sum\":" << stats.weighted_ce_sum
      << ",\"mean_presented_weighted_ce\":"
      << stats.mean_presented_weighted_ce << '}';
  return out.str();
}

std::string GateJson(const OfflineGateResult& gate) {
  std::ostringstream out;
  out << std::setprecision(17)
      << "{\"parent\":{\"all\":" << MetricLevelJson(gate.parent_all)
      << ",\"changed\":" << MetricLevelJson(gate.parent_changed)
      << ",\"preserved\":" << MetricLevelJson(gate.parent_preserved)
      << "},\"control_child\":{\"all\":"
      << MetricLevelJson(gate.control_all) << ",\"changed\":"
      << MetricLevelJson(gate.control_changed) << ",\"preserved\":"
      << MetricLevelJson(gate.control_preserved)
      << "},\"treatment_child\":{\"all\":"
      << MetricLevelJson(gate.treatment_all) << ",\"changed\":"
      << MetricLevelJson(gate.treatment_changed) << ",\"preserved\":"
      << MetricLevelJson(gate.treatment_preserved)
      << "},\"paired\":{\"changed\":" << PairedJson(gate.changed)
      << ",\"preserved\":" << PairedJson(gate.preserved)
      << "},\"changed_by_role\":[";
  for (size_t i = 0; i < gate.roles.size(); ++i) {
    if (i) out << ',';
    out << "{\"role\":" << JsonString(gate.roles[i].role)
        << ",\"paired\":" << PairedJson(gate.roles[i].changed)
        << ",\"lower_mean_changed_ce\":"
        << (gate.roles[i].lower_changed_ce ? "true" : "false") << '}';
  }
  out << "],\"conditions\":{\"changed_fit_and_adoption\":"
      << (gate.condition_1_changed_fit_and_adoption ? "true" : "false")
      << ",\"all_roles_lower_changed_ce\":"
      << (gate.condition_2_all_roles_lower_changed_ce ? "true" : "false")
      << ",\"preserved_noninferiority\":"
      << (gate.condition_3_preserved_noninferiority ? "true" : "false")
      << ",\"integrity_assertions\":"
      << (gate.condition_4_integrity_assertions ? "true" : "false")
      << "},\"passed\":" << (gate.passed ? "true" : "false") << '}';
  return out.str();
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  using namespace open_spiel;
  const std::string mode = absl::GetFlag(FLAGS_mode);
  const bool preflight = mode == "preflight";
  const bool run = mode == "run";
  if (!preflight && !run) SpielFatalError("--mode=preflight|run is required");
  const std::string output_path = absl::GetFlag(FLAGS_output);
  const std::string start_path = absl::GetFlag(FLAGS_start_model);
  if (output_path.empty() || start_path.empty()) {
    SpielFatalError("--output and --start_model are required");
  }
  std::string refusal_error;
  if (!RefuseOfflineOptimizerCheckpoint(
          absl::GetFlag(FLAGS_optimizer_checkpoint), &refusal_error)) {
    SpielFatalError(refusal_error);
  }
  at::globalContext().setDeterministicAlgorithms(true, /*silent=*/true);

  const std::vector<std::string> shard_paths =
      Csv(absl::GetFlag(FLAGS_shards));
  const std::vector<uint64_t> expected_sizes =
      SizeCsv(absl::GetFlag(FLAGS_expected_shard_sizes));
  const std::vector<std::string> expected_shas =
      Csv(absl::GetFlag(FLAGS_expected_shard_sha256));
  if (shard_paths.size() != 10 || expected_sizes.size() != 10 ||
      expected_shas.size() != 10) {
    SpielFatalError("exactly ten shard paths, sizes, and hashes are required");
  }

  const std::string binary_path = std::filesystem::canonical(argv[0]).string();
  size_t binary_size = 0;
  const std::string binary_sha256 =
      ComputeFileSHA256(binary_path, &binary_size);
  if (run && (absl::GetFlag(FLAGS_expected_binary_sha256).empty() ||
              absl::GetFlag(FLAGS_expected_binary_sha256) != binary_sha256)) {
    SpielFatalError("executable SHA-256 differs from the frozen manifest");
  }
  size_t start_size = 0;
  const std::string start_sha256 = ComputeFileSHA256(start_path, &start_size);
  if (absl::GetFlag(FLAGS_expected_start_sha256).empty() ||
      absl::GetFlag(FLAGS_expected_start_sha256) != start_sha256) {
    SpielFatalError("registered start-model file SHA-256 mismatch");
  }

  std::vector<SearchPiRow> rows;
  std::vector<ResolvedShard> resolved;
  resolved.reserve(10);
  std::string combined_target_hash;
  std::string combined_extended_hash;
  for (size_t i = 0; i < shard_paths.size(); ++i) {
    ResolvedShard shard;
    shard.path = shard_paths[i];
    shard.expected_size = expected_sizes[i];
    shard.expected_sha256 = expected_shas[i];
    size_t size = 0;
    shard.actual_sha256 = ComputeFileSHA256(shard.path, &size);
    shard.actual_size = size;
    shard.generation = static_cast<int>(i + 1);
    if (shard.actual_size != shard.expected_size ||
        shard.actual_sha256 != shard.expected_sha256) {
      SpielFatalError("registered shard byte/hash mismatch: " + shard.path);
    }
    std::vector<SearchPiRow> cohort;
    std::string read_error;
    if (!ReadSearchPiRowShard(shard.path, &cohort, &read_error)) {
      SpielFatalError("offline shard read failed for " + shard.path + ": " +
                      read_error);
    }
    shard.rows = static_cast<int64_t>(cohort.size());
    for (size_t ri = 0; ri < cohort.size(); ++ri) {
      const SearchPiRow& row = cohort[ri];
      ValidateParsedRow(row, shard.generation, ri, shard.path);
      shard.target_hash = ChainSearchPiTargetHash(shard.target_hash, row);
      shard.extended_hash =
          ChainSearchPiExtendedRowHash(shard.extended_hash, row);
      combined_target_hash =
          ChainSearchPiTargetHash(combined_target_hash, row);
      combined_extended_hash =
          ChainSearchPiExtendedRowHash(combined_extended_hash, row);
    }
    rows.insert(rows.end(), std::make_move_iterator(cohort.begin()),
                std::make_move_iterator(cohort.end()));
    resolved.push_back(std::move(shard));
  }
  if (static_cast<int64_t>(rows.size()) != kOfflineChangedStateRows) {
    SpielFatalError("combined parsed row count is not 92,379");
  }
  if (!absl::GetFlag(FLAGS_expected_target_hash).empty() &&
      absl::GetFlag(FLAGS_expected_target_hash) != combined_target_hash) {
    SpielFatalError("combined target hash differs from frozen manifest");
  }
  if (!absl::GetFlag(FLAGS_expected_extended_hash).empty() &&
      absl::GetFlag(FLAGS_expected_extended_hash) != combined_extended_hash) {
    SpielFatalError("combined extended-row hash differs from frozen manifest");
  }
  if (run && (absl::GetFlag(FLAGS_expected_target_hash).empty() ||
              absl::GetFlag(FLAGS_expected_extended_hash).empty())) {
    SpielFatalError("run mode requires both frozen combined row hashes");
  }

  const auto& registered_array = RegisteredOfflineRoleSpecs();
  const std::vector<OfflineRoleSpec> registered_specs(registered_array.begin(),
                                                       registered_array.end());
  OfflineWeightPlan weights;
  std::string weight_error;
  if (!BuildOfflineWeightPlan(rows, registered_specs, &weights,
                              &weight_error)) {
    SpielFatalError("offline weight-plan validation failed: " + weight_error);
  }
  const std::vector<OfflineBatchBoundary> boundaries =
      BuildOfflineBatchBoundaries(rows.size(), kOfflineChangedStateBatchSize);
  if (boundaries.size() != 361 || boundaries.back().count != 219) {
    SpielFatalError("registered last-minibatch contract mismatch");
  }
  const std::string control_presentation = OfflinePresentationDigest(
      combined_extended_hash, boundaries, kOfflineChangedStateSeed);
  const std::string treatment_presentation = OfflinePresentationDigest(
      combined_extended_hash, boundaries, kOfflineChangedStateSeed);
  if (control_presentation != treatment_presentation) {
    SpielFatalError("arm presentation contracts differ");
  }

  const torch::Device device =
      ResolveDevice(absl::GetFlag(FLAGS_device), preflight);
  auto control_model = LoadModel(start_path, device);
  auto treatment_model = LoadModel(start_path, device);
  auto control_optimizer = MakeOfflineFreshAdamW(control_model);
  auto treatment_optimizer = MakeOfflineFreshAdamW(treatment_model);
  const std::string observed_start_digest =
      CanonicalOfflineModuleDigest(*control_model);
  const std::string expected_start_digest =
      absl::GetFlag(FLAGS_expected_start_digest).empty()
          ? observed_start_digest
          : absl::GetFlag(FLAGS_expected_start_digest);
  if (run && absl::GetFlag(FLAGS_expected_start_digest).empty()) {
    SpielFatalError("run mode requires the frozen canonical start digest");
  }
  std::string boundary_error;
  if (!ValidateFreshOfflineBoundary(
          *control_model, *control_optimizer, *treatment_model,
          *treatment_optimizer, expected_start_digest, &boundary_error)) {
    SpielFatalError("fresh optimizer boundary failed: " + boundary_error);
  }
  const OfflineAdamWContract initial_optimizer_contract =
      CaptureOfflineAdamWContract(*control_model, *control_optimizer);

  const std::string preflight_json = PreflightJson(
      mode, binary_path, binary_sha256, start_path, start_size, start_sha256,
      observed_start_digest, resolved, rows.size(), combined_target_hash,
      combined_extended_hash, weights, boundaries, control_presentation,
      initial_optimizer_contract);
  if (preflight) {
    WriteAtomically(output_path, preflight_json);
    std::cout << "OFFLINE_PREFLIGHT_OK rows=" << rows.size()
              << " target=" << combined_target_hash
              << " extended=" << combined_extended_hash
              << " start_digest=" << observed_start_digest
              << " output=" << output_path << '\n';
    return 0;
  }

  const std::string output_dir = absl::GetFlag(FLAGS_output_dir);
  if (output_dir.empty()) SpielFatalError("run mode requires --output_dir");
  std::filesystem::create_directories(output_dir);
  const std::string control_path =
      (std::filesystem::path(output_dir) / "control_model.pt").string();
  const std::string treatment_path =
      (std::filesystem::path(output_dir) / "treatment_model.pt").string();

  // Evaluate the shared parent before either update.  Evaluation is read-only;
  // the fresh boundary is re-asserted afterwards, immediately before the first
  // backward pass.
  const OfflineModelMetrics parent = EvaluateOfflineChangedStateRows(
      control_model, rows, boundaries, kObservationSize, kActionDim, device);
  if (!ValidateFreshOfflineBoundary(
          *control_model, *control_optimizer, *treatment_model,
          *treatment_optimizer, expected_start_digest, &boundary_error)) {
    SpielFatalError("fresh boundary changed during parent evaluation: " +
                    boundary_error);
  }

  const OfflineTrainingStats control_training = TrainOfflineChangedStateArm(
      control_model, *control_optimizer, rows, weights.control, boundaries,
      kObservationSize, kActionDim, device, kOfflineChangedStateSeed);
  const OfflineModelMetrics control_child = EvaluateOfflineChangedStateRows(
      control_model, rows, boundaries, kObservationSize, kActionDim, device);
  const std::string control_child_digest =
      CanonicalOfflineModuleDigest(*control_model);
  SaveModelAtomically(control_model, control_path);
  size_t control_model_size = 0;
  const std::string control_model_sha256 =
      ComputeFileSHA256(control_path, &control_model_size);

  // Treatment has not participated in control training.  Its model digest,
  // empty state map, and exact parameter-group contract are asserted again
  // immediately before its first backward pass.
  if (CanonicalOfflineModuleDigest(*treatment_model) !=
          expected_start_digest ||
      !treatment_optimizer->state().empty()) {
    SpielFatalError("treatment fresh boundary changed before first backward");
  }
  if (!OfflineAdamWContractsEqual(
          initial_optimizer_contract,
          CaptureOfflineAdamWContract(*treatment_model, *treatment_optimizer),
          &boundary_error)) {
    SpielFatalError("treatment optimizer contract changed: " + boundary_error);
  }
  torch::manual_seed(kOfflineChangedStateSeed);
  const OfflineTrainingStats treatment_training = TrainOfflineChangedStateArm(
      treatment_model, *treatment_optimizer, rows, weights.treatment,
      boundaries, kObservationSize, kActionDim, device,
      kOfflineChangedStateSeed);
  const OfflineModelMetrics treatment_child = EvaluateOfflineChangedStateRows(
      treatment_model, rows, boundaries, kObservationSize, kActionDim, device);
  const std::string treatment_child_digest =
      CanonicalOfflineModuleDigest(*treatment_model);
  SaveModelAtomically(treatment_model, treatment_path);
  size_t treatment_model_size = 0;
  const std::string treatment_model_sha256 =
      ComputeFileSHA256(treatment_path, &treatment_model_size);

  const bool runtime_integrity =
      control_training.presentations == kOfflineChangedStateRows &&
      treatment_training.presentations == kOfflineChangedStateRows &&
      control_training.minibatches == 361 &&
      treatment_training.minibatches == 361 &&
      control_training.final_minibatch_size == 219 &&
      treatment_training.final_minibatch_size == 219 &&
      std::abs(control_training.presentation_weight_mass -
               kOfflineChangedStateRows) < 1e-8 &&
      std::abs(treatment_training.presentation_weight_mass -
               kOfflineChangedStateRows) < 1e-8;
  if (!runtime_integrity) {
    SpielFatalError("runtime presentation/weight integrity assertion failed");
  }
  const OfflineGateResult gate = ApplyOfflineChangedStateGate(
      rows, parent, control_child, treatment_child,
      /*integrity_assertions_passed=*/true);

  std::ostringstream result;
  result << std::setprecision(17)
         << "{\"schema_version\":1,\"analysis_kind\":"
            "\"FROZEN_OFFLINE_CHANGED_STATE_WEIGHTING_RESULT\""
         << ",\"confirmatory\":false,\"termination_label\":"
         << JsonString(kTerminationLabel) << ",\"preflight\":"
         << preflight_json
         << ",\"device\":" << JsonString(device.is_cuda() ? "cuda" : "cpu")
         << ",\"training\":{\"control\":"
         << TrainingStatsJson(control_training) << ",\"treatment\":"
         << TrainingStatsJson(treatment_training) << '}'
         << ",\"models\":{\"control\":{\"path\":"
         << JsonString(control_path) << ",\"size\":" << control_model_size
         << ",\"sha256\":" << JsonString(control_model_sha256)
         << ",\"canonical_digest\":" << JsonString(control_child_digest)
         << "},\"treatment\":{\"path\":"
         << JsonString(treatment_path) << ",\"size\":"
         << treatment_model_size << ",\"sha256\":"
         << JsonString(treatment_model_sha256)
         << ",\"canonical_digest\":" << JsonString(treatment_child_digest)
         << "}}"
         << ",\"offline_gate\":" << GateJson(gate)
         << ",\"strength_evaluation_authorised\":"
         << (gate.passed ? "true" : "false") << '}';
  WriteAtomically(output_path, result.str());
  std::cout << "OFFLINE_RUN_OK rows=" << rows.size()
            << " gate=" << (gate.passed ? "PASS" : "FAIL")
            << " result=" << output_path << '\n';
  return 0;
}
