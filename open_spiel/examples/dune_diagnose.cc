#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <random>
#include <cmath>
#include <iomanip>

#include "open_spiel/spiel.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include <torch/torch.h>

#include "dune_network.h"

namespace open_spiel {
namespace {

struct DiagnosticStats {
  double total_raw_centered_abs_max = 0.0;
  double total_raw_offset_abs = 0.0;
  double total_std = 0.0;
  double total_range = 0.0;
  double total_top1 = 0.0;
  double total_entropy = 0.0;
  double total_legal_count = 0.0;
  int64_t total_decisions = 0;
};

void RunDiagnostic(const std::string& model_checkpoint,
                   int num_games,
                   int hidden_dim,
                   int num_blocks) {
  std::shared_ptr<const Game> game = LoadGame("dune_imperium");
  int64_t obs_size = game->InformationStateTensorSize();
  int64_t action_size = game->NumDistinctActions();

  torch::InferenceMode inference_guard;
  torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

  auto model = std::make_shared<SharedDunePolicyValueNetImpl>(obs_size, hidden_dim, action_size, num_blocks);
  model->eval();
  try {
    torch::serialize::InputArchive archive;
    archive.load_from(model_checkpoint, device);
    model->load(archive);
  } catch (const c10::Error& e) {
    std::cerr << "Failed to load Model checkpoint: " << model_checkpoint << "\n" << e.msg() << "\n";
    return;
  }
  model->to(device);

  auto sync_mutex = std::make_shared<std::shared_mutex>();
  auto evaluator = std::make_shared<BatchedEvaluator>(
      model, 1, 1, device, sync_mutex.get(), 0.0f);

  std::random_device rd;
  std::mt19937 rng(rd());

  DiagnosticStats stats;
  std::vector<float> obs(obs_size, 0.0f);

  for (int g = 0; g < num_games; ++g) {
    std::unique_ptr<State> state = game->NewInitialState();
    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        std::vector<std::pair<Action, double>> outcomes = state->ChanceOutcomes();
        Action action;
        if (game->GetType().chance_mode == GameType::ChanceMode::kSampledStochastic) {
          action = outcomes.front().first;
        } else {
          action = SampleAction(outcomes, rng).first;
        }
        state->ApplyAction(action);
        continue;
      }

      Player current_player = state->CurrentPlayer();
      std::vector<Action> legal_actions = state->LegalActions();
      if (legal_actions.empty()) {
        break;
      }

      // Fill observation buffer
      std::fill(obs.begin(), obs.end(), 0.0f);
      state->InformationStateTensor(current_player, absl::MakeSpan(obs));

      // Get policy predictions
      EvalResult result = evaluator->Evaluate(obs);

      double raw_sum = 0.0;
      for (Action a : legal_actions) {
        raw_sum += result.logits[a];
      }
      double raw_mean = raw_sum / legal_actions.size();
      double raw_centered_abs_max = 0.0;
      for (Action a : legal_actions) {
        raw_centered_abs_max = std::max(raw_centered_abs_max,
                                        std::abs(static_cast<double>(result.logits[a]) - raw_mean));
      }

      CenterAndCapLegalLogits(result.logits, legal_actions, 10.0f);

      // Collect diagnostic metrics over legal actions
      std::vector<float> legal_logits;
      legal_logits.reserve(legal_actions.size());
      float max_logit = -1e9f;
      for (Action a : legal_actions) {
        float l = result.logits[a];
        legal_logits.push_back(l);
        if (l > max_logit) {
          max_logit = l;
        }
      }

      // Softmax over legal actions
      double sum_exp = 0.0;
      std::vector<double> probs(legal_actions.size(), 0.0);
      for (size_t i = 0; i < legal_logits.size(); ++i) {
        double exp_val = std::exp(legal_logits[i] - max_logit);
        probs[i] = exp_val;
        sum_exp += exp_val;
      }

      double top1_prob = 0.0;
      double entropy = 0.0;
      for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] /= sum_exp;
        if (probs[i] > top1_prob) {
          top1_prob = probs[i];
        }
        if (probs[i] > 1e-12) {
          entropy -= probs[i] * std::log(probs[i]);
        }
      }

      // Compute mean, std, range of legal logits
      double sum_logits = 0.0;
      float min_logit = 1e9f;
      float actual_max_logit = -1e9f;
      for (float l : legal_logits) {
        sum_logits += l;
        if (l < min_logit) min_logit = l;
        if (l > actual_max_logit) actual_max_logit = l;
      }
      double mean_logit = sum_logits / legal_logits.size();

      double sum_sq_diff = 0.0;
      for (float l : legal_logits) {
        sum_sq_diff += (l - mean_logit) * (l - mean_logit);
      }
      double std_logit = std::sqrt(sum_sq_diff / legal_logits.size());
      double range_logit = actual_max_logit - min_logit;

      // Accumulate
      stats.total_raw_centered_abs_max += raw_centered_abs_max;
      stats.total_raw_offset_abs += std::abs(raw_mean);
      stats.total_std += std_logit;
      stats.total_range += range_logit;
      stats.total_top1 += top1_prob;
      stats.total_entropy += entropy;
      stats.total_legal_count += legal_logits.size();
      stats.total_decisions++;

      // Greedy choice to advance self-play
      Action chosen_action = legal_actions.front();
      float best_l = -1e9f;
      for (Action a : legal_actions) {
        if (result.logits[a] > best_l) {
          best_l = result.logits[a];
          chosen_action = a;
        }
      }

      state->ApplyAction(chosen_action);
    }
  }

  if (stats.total_decisions > 0) {
    double avg_std = stats.total_std / stats.total_decisions;
    double avg_range = stats.total_range / stats.total_decisions;
    double avg_top1 = stats.total_top1 / stats.total_decisions;
    double avg_entropy = stats.total_entropy / stats.total_decisions;
    double avg_legal = stats.total_legal_count / stats.total_decisions;
    double avg_raw_centered_abs_max = stats.total_raw_centered_abs_max / stats.total_decisions;
    double avg_raw_offset_abs = stats.total_raw_offset_abs / stats.total_decisions;

    std::filesystem::path path(model_checkpoint);
    std::string filename = path.filename().string();

    std::cout << absl::StrFormat("%-35s | %8d | %9.4f | %9.4f | %8.4f | %10.4f | %10.4f | %8.4f | %10.2f",
                                 filename, stats.total_decisions, avg_raw_centered_abs_max,
                                 avg_raw_offset_abs, avg_std, avg_range, avg_top1,
                                 avg_entropy, avg_legal)
              << std::endl;
  }
}

} // namespace
} // namespace open_spiel

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: dune_diagnose <checkpoint1.pt> [checkpoint2.pt ...]\n";
    return 1;
  }

  int hidden_dim = 2048;
  int num_blocks = 8;
  int num_games = 5;

  std::cout << absl::StrFormat("%-35s | %8s | %9s | %9s | %8s | %10s | %10s | %8s | %10s",
                               "Checkpoint", "Decisions", "RawCtrMax", "RawOffset",
                               "Cap Std", "Cap Range", "Avg Top1", "Entropy", "Avg Legal")
            << std::endl;
  std::cout << std::string(134, '-') << std::endl;

  for (int i = 1; i < argc; ++i) {
    std::string checkpoint = argv[i];
    open_spiel::RunDiagnostic(checkpoint, num_games, hidden_dim, num_blocks);
  }

  return 0;
}
