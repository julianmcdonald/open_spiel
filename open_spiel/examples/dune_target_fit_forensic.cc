// Descriptive target-fit forensic for Search-PI Rung-4N.
//
// This is deliberately a separate executable from the frozen trainer.  It
// reconstructs the exact rows presented to one completed F or W update,
// verifies the trainer's recorded target/extended row hashes, and measures the
// search-target CE/KL and teacher-argmax adoption before and after that update.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/spiel_utils.h"

#include <torch/torch.h>

#include "dune_network.h"
#include "dune_ppo_training_utils.h"
#include "dune_search_pi.h"
#include "dune_search_pi_replay.h"

ABSL_FLAG(std::string, arm, "", "F or W.");
ABSL_FLAG(int, generation, 0, "Completed update generation.");
ABSL_FLAG(std::string, shards, "", "Comma-separated cohort shards, oldest first.");
ABSL_FLAG(int, sample_count, 0, "Exact W sample count; ignored for F.");
ABSL_FLAG(uint64_t, sample_seed, 0, "Exact W replay sample seed.");
ABSL_FLAG(std::string, parent_model, "", "Checkpoint before the update.");
ABSL_FLAG(std::string, child_model, "", "Checkpoint after the update.");
ABSL_FLAG(std::string, expected_target_hash, "", "Recorded trained target hash.");
ABSL_FLAG(std::string, expected_extended_hash, "", "Recorded trained extended-row hash.");
ABSL_FLAG(std::string, output, "", "Output JSON path.");
ABSL_FLAG(int, observation_size, 5580, "Model input width.");
ABSL_FLAG(int, action_dim, 2391, "Policy vocabulary size.");
ABSL_FLAG(int, hidden_dim, 2048, "Model hidden width.");
ABSL_FLAG(int, num_blocks, 8, "Residual block count.");
ABSL_FLAG(int, batch_size, 256, "Inference batch size.");
ABSL_FLAG(double, logit_cap, 10.0, "Learner's legal-centered tanh logit cap.");

// Link-only definitions required by dune_ppo_training_utils.cc. The forensic
// calls only CenterAndCapLogitsTensor; none of these flags is read on its path.
ABSL_FLAG(int, ppo_minibatch_size, 2048, "INERT here (link satisfaction).");
ABSL_FLAG(int, ppo_update_epochs, 4, "INERT here (link satisfaction).");
ABSL_FLAG(double, ppo_clip_epsilon, 0.2, "INERT here (link satisfaction).");
ABSL_FLAG(bool, normalize_advantages, true, "INERT here (link satisfaction).");
ABSL_FLAG(bool, ppo_clip_value_loss, true, "INERT here (link satisfaction).");
ABSL_FLAG(double, entropy_coef, 0.01, "INERT here (link satisfaction).");
ABSL_FLAG(double, value_coef, 0.5, "INERT here (link satisfaction).");
ABSL_FLAG(double, target_kl, 0.0, "INERT here (link satisfaction).");
ABSL_FLAG(bool, train_amp, true, "INERT here (link satisfaction).");
ABSL_FLAG(double, grad_clip_norm, 0.5, "INERT here (link satisfaction).");
ABSL_FLAG(uint64_t, shaping_start_env_steps, 206830543,
          "INERT here (link satisfaction).");
ABSL_FLAG(uint64_t, shaping_decay_env_steps, 0,
          "INERT here (link satisfaction).");
ABSL_FLAG(bool, diagnostics_only, false, "INERT here (link satisfaction).");

namespace open_spiel {
namespace {

constexpr double kZ95 = 1.959963984540054;

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

const char* RoleName(DuneDecisionRole role) {
  switch (role) {
    case DuneDecisionRole::kForcedOrBookkeeping: return "forced_or_bookkeeping";
    case DuneDecisionRole::kLeaderSelection: return "leader_selection";
    case DuneDecisionRole::kAgentPrimary: return "agent_primary";
    case DuneDecisionRole::kAgentContinuation: return "agent_continuation";
    case DuneDecisionRole::kPurchase: return "purchase";
    case DuneDecisionRole::kCombatIntrigue: return "combat_intrigue";
    case DuneDecisionRole::kOtherOptional: return "other_optional";
  }
  return "invalid";
}

struct ModelMetrics {
  std::vector<double> ce;
  std::vector<double> kl;
  std::vector<uint8_t> adopted;
};

std::shared_ptr<SharedDunePolicyValueNetImpl> LoadModel(
    const std::string& path, torch::Device device, int64_t obs_size,
    int64_t action_dim, int64_t hidden_dim, int num_blocks) {
  auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
      obs_size, hidden_dim, action_dim, num_blocks, /*use_nonlinear=*/false);
  model->eval();
  torch::serialize::InputArchive archive;
  archive.load_from(path, device);
  model->load(archive);
  model->to(device);
  return model;
}

ModelMetrics EvaluateRows(const std::vector<SearchPiRow>& rows,
                          const std::string& checkpoint,
                          torch::Device device, int64_t obs_size,
                          int64_t action_dim, int64_t hidden_dim,
                          int num_blocks, int64_t batch_size,
                          float logit_cap) {
  auto model = LoadModel(checkpoint, device, obs_size, action_dim, hidden_dim,
                         num_blocks);
  torch::InferenceMode inference_guard;
  ModelMetrics out;
  out.ce.reserve(rows.size());
  out.kl.reserve(rows.size());
  out.adopted.reserve(rows.size());

  const auto f32 = torch::TensorOptions().dtype(torch::kFloat32);
  const auto boolopt = torch::TensorOptions().dtype(torch::kBool);
  for (int64_t start = 0; start < static_cast<int64_t>(rows.size());
       start += batch_size) {
    const int64_t count = std::min<int64_t>(
        batch_size, static_cast<int64_t>(rows.size()) - start);
    torch::Tensor states = torch::zeros({count, obs_size}, f32);
    torch::Tensor masks = torch::zeros({count, action_dim}, boolopt);
    torch::Tensor targets = torch::zeros({count, action_dim}, f32);
    float* states_ptr = states.data_ptr<float>();
    bool* masks_ptr = masks.data_ptr<bool>();
    float* targets_ptr = targets.data_ptr<float>();
    for (int64_t bi = 0; bi < count; ++bi) {
      const SearchPiRow& row = rows[start + bi];
      if (row.observation.empty() ||
          static_cast<int64_t>(row.observation.size()) > obs_size ||
          row.legal_actions.empty() ||
          row.legal_actions.size() != row.target_probs.size()) {
        SpielFatalError("malformed forensic row at presentation index " +
                        std::to_string(start + bi));
      }
      std::copy(row.observation.begin(), row.observation.end(),
                states_ptr + bi * obs_size);
      for (size_t j = 0; j < row.legal_actions.size(); ++j) {
        const Action action = row.legal_actions[j];
        if (action < 0 || action >= action_dim) {
          SpielFatalError("out-of-range legal action in forensic row");
        }
        masks_ptr[bi * action_dim + action] = true;
        targets_ptr[bi * action_dim + action] =
            static_cast<float>(row.target_probs[j]);
      }
    }

    states = states.to(device);
    masks = masks.to(device);
    targets = targets.to(device);
    const auto outputs = model->forward(states);
    const torch::Tensor centered = CenterAndCapLogitsTensor(
        outputs.logits, masks, logit_cap);
    const torch::Tensor masked =
        centered.masked_fill(masks.logical_not(), -1e9f);
    const torch::Tensor logp = torch::log_softmax(masked, -1);
    const torch::Tensor ce = -(targets * logp).sum(-1);
    const torch::Tensor entropy_terms = torch::where(
        targets > 0, targets * torch::log(targets), torch::zeros_like(targets));
    const torch::Tensor kl = ce + entropy_terms.sum(-1);
    const torch::Tensor adopted =
        torch::argmax(masked, -1).eq(torch::argmax(targets, -1));

    const torch::Tensor ce_cpu = ce.to(
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kDouble));
    const torch::Tensor kl_cpu = kl.to(
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kDouble));
    const torch::Tensor adopted_cpu = adopted.to(
        torch::TensorOptions().device(torch::kCPU).dtype(torch::kUInt8));
    const double* ce_ptr = ce_cpu.data_ptr<double>();
    const double* kl_ptr = kl_cpu.data_ptr<double>();
    const uint8_t* adopted_ptr = adopted_cpu.data_ptr<uint8_t>();
    for (int64_t bi = 0; bi < count; ++bi) {
      out.ce.push_back(ce_ptr[bi]);
      out.kl.push_back(kl_ptr[bi]);
      out.adopted.push_back(adopted_ptr[bi]);
    }
  }
  model.reset();
  return out;
}

struct Summary {
  int64_t n = 0;
  double parent_ce_sum = 0.0;
  double child_ce_sum = 0.0;
  double parent_kl_sum = 0.0;
  double child_kl_sum = 0.0;
  double reduction_sum = 0.0;
  double reduction_sq_sum = 0.0;
  int64_t parent_adopted = 0;
  int64_t child_adopted = 0;
  int64_t newly_adopted = 0;
  int64_t lost_adoption = 0;

  void Add(double parent_ce, double child_ce, double parent_kl,
           double child_kl, bool parent_adopt, bool child_adopt) {
    const double reduction = parent_ce - child_ce;
    ++n;
    parent_ce_sum += parent_ce;
    child_ce_sum += child_ce;
    parent_kl_sum += parent_kl;
    child_kl_sum += child_kl;
    reduction_sum += reduction;
    reduction_sq_sum += reduction * reduction;
    parent_adopted += parent_adopt;
    child_adopted += child_adopt;
    newly_adopted += (!parent_adopt && child_adopt);
    lost_adoption += (parent_adopt && !child_adopt);
  }
};

std::pair<double, double> Wilson(int64_t success, int64_t n) {
  if (n == 0) return {0.0, 0.0};
  const double nn = static_cast<double>(n);
  const double p = static_cast<double>(success) / nn;
  const double z2 = kZ95 * kZ95;
  const double denom = 1.0 + z2 / nn;
  const double center = (p + z2 / (2.0 * nn)) / denom;
  const double margin = kZ95 *
      std::sqrt((p * (1.0 - p) + z2 / (4.0 * nn)) / nn) / denom;
  return {std::max(0.0, center - margin),
          std::min(1.0, center + margin)};
}

std::string SummaryJson(const Summary& s) {
  std::ostringstream out;
  out << std::setprecision(17);
  out << "{\"n\":" << s.n;
  if (s.n == 0) {
    out << ",\"measured\":false}";
    return out.str();
  }
  const double n = static_cast<double>(s.n);
  const double reduction_mean = s.reduction_sum / n;
  double reduction_se = 0.0;
  if (s.n > 1) {
    const double variance = std::max(
        0.0, (s.reduction_sq_sum - n * reduction_mean * reduction_mean) /
                 static_cast<double>(s.n - 1));
    reduction_se = std::sqrt(variance / n);
  }
  const auto parent_ci = Wilson(s.parent_adopted, s.n);
  const auto child_ci = Wilson(s.child_adopted, s.n);
  out << ",\"measured\":true"
      << ",\"parent_ce_mean\":" << s.parent_ce_sum / n
      << ",\"child_ce_mean\":" << s.child_ce_sum / n
      << ",\"target_ce_reduction_mean\":" << reduction_mean
      << ",\"target_ce_reduction_sd\":" << reduction_se * std::sqrt(n)
      << ",\"target_ce_reduction_se\":" << reduction_se
      << ",\"target_ce_reduction_ci95\":["
      << reduction_mean - kZ95 * reduction_se << ','
      << reduction_mean + kZ95 * reduction_se << ']'
      << ",\"parent_kl_mean\":" << s.parent_kl_sum / n
      << ",\"child_kl_mean\":" << s.child_kl_sum / n
      << ",\"target_kl_reduction_mean\":"
      << (s.parent_kl_sum - s.child_kl_sum) / n
      << ",\"parent_teacher_argmax_adoption\":"
      << static_cast<double>(s.parent_adopted) / n
      << ",\"parent_teacher_argmax_adopted_count\":" << s.parent_adopted
      << ",\"parent_teacher_argmax_adoption_ci95\":[" << parent_ci.first
      << ',' << parent_ci.second << ']'
      << ",\"child_teacher_argmax_adoption\":"
      << static_cast<double>(s.child_adopted) / n
      << ",\"child_teacher_argmax_adopted_count\":" << s.child_adopted
      << ",\"child_teacher_argmax_adoption_ci95\":[" << child_ci.first
      << ',' << child_ci.second << ']'
      << ",\"teacher_argmax_adoption_change\":"
      << static_cast<double>(s.child_adopted - s.parent_adopted) / n
      << ",\"newly_adopted\":" << s.newly_adopted
      << ",\"lost_adoption\":" << s.lost_adoption << '}';
  return out.str();
}

void WriteAtomically(const std::string& path, const std::string& content) {
  const std::filesystem::path target(path);
  std::filesystem::create_directories(target.parent_path());
  const std::filesystem::path tmp = target.string() + ".tmp";
  {
    std::ofstream out(tmp, std::ios::out | std::ios::trunc);
    if (!out) SpielFatalError("cannot open forensic temp output: " + tmp.string());
    out << content << '\n';
    out.flush();
    if (!out) SpielFatalError("cannot write forensic temp output: " + tmp.string());
  }
  std::filesystem::rename(tmp, target);
}

}  // namespace
}  // namespace open_spiel

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  using namespace open_spiel;
  const std::string arm = absl::GetFlag(FLAGS_arm);
  const int generation = absl::GetFlag(FLAGS_generation);
  const std::string parent_path = absl::GetFlag(FLAGS_parent_model);
  const std::string child_path = absl::GetFlag(FLAGS_child_model);
  const std::string output_path = absl::GetFlag(FLAGS_output);
  if ((arm != "F" && arm != "W") || generation <= 0 ||
      parent_path.empty() || child_path.empty() || output_path.empty()) {
    SpielFatalError("--arm=F|W, positive --generation, --parent_model, "
                    "--child_model and --output are required");
  }
  const std::vector<std::string> shard_paths =
      absl::StrSplit(absl::GetFlag(FLAGS_shards), ',');
  if (shard_paths.empty() || (shard_paths.size() == 1 && shard_paths[0].empty())) {
    SpielFatalError("at least one --shards path is required");
  }

  std::vector<std::vector<SearchPiRow>> window;
  window.reserve(shard_paths.size());
  int64_t window_rows = 0;
  for (const std::string& path : shard_paths) {
    std::vector<SearchPiRow> cohort;
    std::string error;
    if (!ReadSearchPiRowShard(path, &cohort, &error)) {
      SpielFatalError("forensic shard read failed for " + path + ": " + error);
    }
    window_rows += static_cast<int64_t>(cohort.size());
    window.push_back(std::move(cohort));
  }

  std::vector<SearchPiRow> rows;
  if (arm == "F") {
    if (window.size() != 1) {
      SpielFatalError("F forensic requires exactly its fresh cohort shard");
    }
    rows = window.front();
  } else {
    const int sample_count = absl::GetFlag(FLAGS_sample_count);
    if (sample_count <= 0 || sample_count > window_rows) {
      SpielFatalError("W forensic sample count is not positive or exceeds window");
    }
    rows = SampleUniformReplayWindow(window, static_cast<size_t>(sample_count),
                                     absl::GetFlag(FLAGS_sample_seed));
  }

  std::string target_hash;
  std::string extended_hash;
  std::map<int, int64_t> source_generation_counts;
  for (const SearchPiRow& row : rows) {
    target_hash = ChainSearchPiTargetHash(target_hash, row);
    extended_hash = ChainSearchPiExtendedRowHash(extended_hash, row);
    source_generation_counts[row.generation]++;
  }
  const std::string expected_target =
      absl::GetFlag(FLAGS_expected_target_hash);
  const std::string expected_extended =
      absl::GetFlag(FLAGS_expected_extended_hash);
  if (expected_target.empty() || target_hash != expected_target) {
    SpielFatalError("exact-row target hash mismatch: got " + target_hash +
                    " expected " + expected_target);
  }
  if (expected_extended.empty() || extended_hash != expected_extended) {
    SpielFatalError("exact-row extended hash mismatch: got " + extended_hash +
                    " expected " + expected_extended);
  }

  const torch::Device device = torch::cuda::is_available()
                                   ? torch::Device(torch::kCUDA)
                                   : torch::Device(torch::kCPU);
  const int64_t obs_size = absl::GetFlag(FLAGS_observation_size);
  const int64_t action_dim = absl::GetFlag(FLAGS_action_dim);
  const int64_t hidden_dim = absl::GetFlag(FLAGS_hidden_dim);
  const int num_blocks = absl::GetFlag(FLAGS_num_blocks);
  const int64_t batch_size = absl::GetFlag(FLAGS_batch_size);
  if (batch_size <= 0) SpielFatalError("--batch_size must be positive");
  const float logit_cap = static_cast<float>(absl::GetFlag(FLAGS_logit_cap));

  const ModelMetrics parent = EvaluateRows(
      rows, parent_path, device, obs_size, action_dim, hidden_dim, num_blocks,
      batch_size, logit_cap);
  const ModelMetrics child = EvaluateRows(
      rows, child_path, device, obs_size, action_dim, hidden_dim, num_blocks,
      batch_size, logit_cap);

  Summary overall;
  std::map<std::string, Summary> by_role;
  std::map<std::string, Summary> by_change;
  std::map<std::string, Summary> by_role_change;
  for (size_t i = 0; i < rows.size(); ++i) {
    const bool p_adopt = parent.adopted[i] != 0;
    const bool c_adopt = child.adopted[i] != 0;
    const auto add = [&](Summary* summary) {
      summary->Add(parent.ce[i], child.ce[i], parent.kl[i], child.kl[i],
                   p_adopt, c_adopt);
    };
    add(&overall);
    const std::string role = RoleName(rows[i].role);
    const std::string change = rows[i].target_argmax_differs_from_raw
                                   ? "search_changed_raw_argmax"
                                   : "search_preserved_raw_argmax";
    add(&by_role[role]);
    add(&by_change[change]);
    add(&by_role_change[role + "/" + change]);
  }

  std::ostringstream json;
  json << std::setprecision(17);
  json << "{\"schema_version\":1"
       << ",\"analysis_kind\":\"DESCRIPTIVE_TARGET_FIT_FORENSIC\""
       << ",\"confirmatory\":false"
       << ",\"arm\":" << JsonString(arm)
       << ",\"generation\":" << generation
       << ",\"device\":" << JsonString(device.is_cuda() ? "cuda" : "cpu")
       << ",\"learner_transform\":{\"legal_centering\":true,"
          "\"tanh_logit_cap\":" << logit_cap
       << ",\"illegal_logit_fill\":-1000000000}"
       << ",\"provenance\":{\"shards\":[";
  for (size_t i = 0; i < shard_paths.size(); ++i) {
    if (i) json << ',';
    json << JsonString(shard_paths[i]);
  }
  json << "],\"window_rows\":" << window_rows
       << ",\"presented_rows\":" << rows.size()
       << ",\"sample_count\":"
       << (arm == "W" ? absl::GetFlag(FLAGS_sample_count)
                       : static_cast<int>(rows.size()))
       << ",\"sample_seed\":"
       << (arm == "W" ? absl::GetFlag(FLAGS_sample_seed) : 0)
       << ",\"target_hash\":" << JsonString(target_hash)
       << ",\"extended_hash\":" << JsonString(extended_hash)
       << ",\"hashes_match_generation_complete\":true"
       << ",\"source_generation_counts\":{";
  bool first = true;
  for (const auto& item : source_generation_counts) {
    if (!first) json << ',';
    first = false;
    json << JsonString(std::to_string(item.first)) << ':' << item.second;
  }
  json << "}}"
       << ",\"parent_model\":" << JsonString(parent_path)
       << ",\"child_model\":" << JsonString(child_path)
       << ",\"overall\":" << SummaryJson(overall);
  const auto emit_map = [&](const char* key,
                            const std::map<std::string, Summary>& values) {
    json << ",\"" << key << "\":{";
    bool first_value = true;
    for (const auto& item : values) {
      if (!first_value) json << ',';
      first_value = false;
      json << JsonString(item.first) << ':' << SummaryJson(item.second);
    }
    json << '}';
  };
  emit_map("by_role", by_role);
  emit_map("by_search_changed_vs_preserved", by_change);
  emit_map("by_role_and_search_changed", by_role_change);
  json << '}';

  WriteAtomically(output_path, json.str());
  std::cout << "FORENSIC_OK arm=" << arm << " generation=" << generation
            << " rows=" << rows.size() << " output=" << output_path << '\n';
  return 0;
}
