#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "dune_network.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/utils/json.h"

ABSL_FLAG(std::string, baseline_checkpoint, "", "Frozen baseline checkpoint.");
ABSL_FLAG(std::string, candidate_checkpoint, "", "Candidate checkpoint.");
ABSL_FLAG(std::string, corpus_path, "data/dune_shadow_corpus.json",
          "Held-out corpus used for legal-logit identity checks.");
ABSL_FLAG(int, hidden_dim, 2048, "Network hidden dimension.");
ABSL_FLAG(int, num_blocks, 8, "Network residual block count.");
ABSL_FLAG(bool, baseline_nonlinear_value_head, false,
          "Whether the baseline uses the nonlinear value head.");
ABSL_FLAG(bool, candidate_nonlinear_value_head, false,
          "Whether the candidate uses the nonlinear value head.");
ABSL_FLAG(int, held_out_states, 10,
          "Number of corpus states used for legal-logit comparison.");

namespace {

using TensorMap = std::unordered_map<std::string, torch::Tensor>;

TensorMap NonValueTensors(
    const torch::OrderedDict<std::string, torch::Tensor>& tensors) {
  TensorMap result;
  for (const auto& item : tensors) {
    if (item.key().rfind("value_head", 0) != 0) {
      result.emplace(item.key(), item.value());
    }
  }
  return result;
}

bool EqualTensorMaps(const TensorMap& baseline, const TensorMap& candidate,
                     const std::string& label) {
  if (baseline.size() != candidate.size()) {
    std::cerr << label << " count mismatch: baseline=" << baseline.size()
              << ", candidate=" << candidate.size() << "\n";
    return false;
  }
  for (const auto& [name, baseline_tensor] : baseline) {
    auto it = candidate.find(name);
    if (it == candidate.end()) {
      std::cerr << "Candidate is missing " << label << ": " << name << "\n";
      return false;
    }
    const torch::Tensor& candidate_tensor = it->second;
    if (baseline_tensor.sizes() != candidate_tensor.sizes() ||
        baseline_tensor.scalar_type() != candidate_tensor.scalar_type() ||
        !torch::equal(baseline_tensor, candidate_tensor)) {
      std::cerr << "Non-value " << label << " differs: " << name << "\n";
      return false;
    }
  }
  return true;
}

struct HeldOutState {
  std::vector<float> observation;
  std::vector<open_spiel::Action> legal_actions;
};

std::vector<HeldOutState> LoadHeldOutStates(const std::string& path,
                                             int limit) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Unable to open held-out corpus: " + path);
  }
  std::string text((std::istreambuf_iterator<char>(input)),
                   std::istreambuf_iterator<char>());
  auto parsed = open_spiel::json::FromString(text);
  if (!parsed || !parsed.value().IsArray()) {
    throw std::runtime_error("Held-out corpus is not a JSON array: " + path);
  }

  std::vector<HeldOutState> states;
  for (const auto& item : parsed.value().GetArray()) {
    auto object = item.GetObject();
    auto observation_it = object.find("observation");
    auto legal_it = object.find("legal_actions");
    if (observation_it == object.end() || legal_it == object.end()) continue;
    HeldOutState state;
    for (const auto& value : observation_it->second.GetArray()) {
      state.observation.push_back(static_cast<float>(value.GetDouble()));
    }
    for (const auto& value : legal_it->second.GetArray()) {
      state.legal_actions.push_back(
          static_cast<open_spiel::Action>(value.GetInt()));
    }
    if (!state.observation.empty() && !state.legal_actions.empty()) {
      states.push_back(std::move(state));
      if (static_cast<int>(states.size()) >= limit) break;
    }
  }
  if (states.empty()) {
    throw std::runtime_error("Held-out corpus contains no usable states.");
  }
  return states;
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const std::string baseline_path =
      absl::GetFlag(FLAGS_baseline_checkpoint);
  const std::string candidate_path =
      absl::GetFlag(FLAGS_candidate_checkpoint);
  if (baseline_path.empty() || candidate_path.empty()) {
    std::cerr << "Both --baseline_checkpoint and --candidate_checkpoint are required.\n";
    return 2;
  }

  try {
    at::set_num_threads(1);
    at::set_num_interop_threads(1);
    torch::NoGradGuard no_grad;

    auto held_out = LoadHeldOutStates(
        absl::GetFlag(FLAGS_corpus_path),
        std::max(1, absl::GetFlag(FLAGS_held_out_states)));
    const int64_t observation_size = held_out.front().observation.size();
    int64_t action_size = 0;
    for (const auto& state : held_out) {
      if (static_cast<int64_t>(state.observation.size()) != observation_size) {
        throw std::runtime_error("Held-out observations have inconsistent sizes.");
      }
      for (open_spiel::Action action : state.legal_actions) {
        action_size = std::max<int64_t>(action_size, action + 1);
      }
    }
    // Dune currently has 2,391 distinct actions. The corpus need not exercise
    // the highest ID, so preserve the model's locked output dimension.
    action_size = std::max<int64_t>(action_size, 2391);

    auto baseline =
        std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
            observation_size, absl::GetFlag(FLAGS_hidden_dim), action_size,
            absl::GetFlag(FLAGS_num_blocks),
            absl::GetFlag(FLAGS_baseline_nonlinear_value_head));
    auto candidate =
        std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
            observation_size, absl::GetFlag(FLAGS_hidden_dim), action_size,
            absl::GetFlag(FLAGS_num_blocks),
            absl::GetFlag(FLAGS_candidate_nonlinear_value_head));
    torch::load(baseline, baseline_path, torch::kCPU);
    torch::load(candidate, candidate_path, torch::kCPU);
    baseline->eval();
    candidate->eval();

    const bool parameters_equal = EqualTensorMaps(
        NonValueTensors(baseline->named_parameters(true)),
        NonValueTensors(candidate->named_parameters(true)), "parameter");
    const bool buffers_equal = EqualTensorMaps(
        NonValueTensors(baseline->named_buffers(true)),
        NonValueTensors(candidate->named_buffers(true)), "buffer");

    std::vector<torch::Tensor> observations;
    observations.reserve(held_out.size());
    for (const auto& state : held_out) {
      observations.push_back(torch::from_blob(
          const_cast<float*>(state.observation.data()), {observation_size},
          torch::TensorOptions().dtype(torch::kFloat32)).clone());
    }
    auto batch = torch::stack(observations);
    auto baseline_logits = baseline->forward(batch).logits;
    auto candidate_logits = candidate->forward(batch).logits;
    bool legal_logits_equal = true;
    double kl_sum = 0.0;
    double max_abs_logit_delta = 0.0;
    for (size_t state_index = 0; state_index < held_out.size(); ++state_index) {
      auto action_indices = torch::tensor(
          held_out[state_index].legal_actions,
          torch::TensorOptions().dtype(torch::kInt64));
      auto baseline_legal = baseline_logits.index(
          {static_cast<int64_t>(state_index)}).index_select(0, action_indices);
      auto candidate_legal = candidate_logits.index(
          {static_cast<int64_t>(state_index)}).index_select(0, action_indices);
      auto baseline_log_probs = torch::log_softmax(baseline_legal, 0);
      auto candidate_log_probs = torch::log_softmax(candidate_legal, 0);
      kl_sum += (torch::softmax(baseline_legal, 0) *
                 (baseline_log_probs - candidate_log_probs)).sum().item<double>();
      max_abs_logit_delta = std::max(
          max_abs_logit_delta,
          (baseline_legal - candidate_legal).abs().max().item<double>());
      for (open_spiel::Action action : held_out[state_index].legal_actions) {
        auto baseline_value = baseline_logits.index(
            {static_cast<int64_t>(state_index), action});
        auto candidate_value = candidate_logits.index(
            {static_cast<int64_t>(state_index), action});
        if (!torch::equal(baseline_value, candidate_value)) {
          legal_logits_equal = false;
        }
      }
    }

    std::cout << "Policy drift summary: mean_legal_kl="
              << kl_sum / held_out.size()
              << ", max_abs_legal_logit_delta=" << max_abs_logit_delta
              << ", held_out_states=" << held_out.size() << ".\n";

    if (parameters_equal && buffers_equal && legal_logits_equal) {
      std::cout << "PASS: non-value parameters, buffers, and held-out legal-action "
                   "logits are bit-for-bit identical across "
                << held_out.size() << " states.\n";
      return 0;
    }
    std::cerr << "Policy identity failed; drift metrics were emitted for the "
                 "required strength-evaluation contingency.\n";
    return 1;
  } catch (const std::exception& error) {
    std::cerr << "Checkpoint identity validation failed: " << error.what()
              << "\n";
    return 2;
  }
}
