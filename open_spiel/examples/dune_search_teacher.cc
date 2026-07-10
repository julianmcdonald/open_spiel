#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <random>
#include <fstream>
#include <filesystem>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <unistd.h>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "dune_puct_is_mcts.h"
#include "dune_evaluator.h"
#include "dune_seed_utils.h"


ABSL_FLAG(std::string, model_checkpoint, "", "Frozen PPO checkpoint");
ABSL_FLAG(std::string, output_dir, "", "Label output directory");
ABSL_FLAG(int, target_labels, 10000, "Stop after this many accepted labels");
ABSL_FLAG(int, max_games, 5000, "Hard game cap");
ABSL_FLAG(int, max_simulations, 50, "MCTS budget");
ABSL_FLAG(double, puct_c, 1.0, "Exploration constant");
ABSL_FLAG(double, value_scale, 1.0, "Leaf value scaling");
ABSL_FLAG(double, blueprint_temperature, 1.0, "PPO action sampling temperature");
ABSL_FLAG(double, search_fraction, 0.10, "Search prob for high-entropy states");
ABSL_FLAG(double, min_entropy_ratio, 0.30, "Entropy threshold");
ABSL_FLAG(double, uniform_ratio, 0.30, "Relative rate for low-entropy states");
ABSL_FLAG(double, target_teacher_kl, 0.10, "Target KL for eta calibration");
ABSL_FLAG(double, eta_max, 50.0, "Maximum eta cap");
ABSL_FLAG(int, min_coverage, 3, "Min covered actions to emit");
ABSL_FLAG(int, min_visits_per_action, 2, "Visits for coverage");
ABSL_FLAG(double, min_prior_mass, 0.50, "Min PPO prior mass on covered actions");
ABSL_FLAG(int, threads, 2, "Game threads");
ABSL_FLAG(int, labels_per_file, 4096, "Labels per output file");
ABSL_FLAG(double, search_opponent_temperature, 0.0,
          "Opponent temperature inside MCTS search. 0.0=greedy, 1.0=policy-consistent.");

ABSL_FLAG(int, hidden_dim, 2048, "Network hidden dimension");
ABSL_FLAG(int, num_blocks, 8, "Network residual block count");
ABSL_FLAG(int, seed, 42, "Seed for rng");


namespace open_spiel {
namespace {

uint64_t ComputeFileHash(const std::string& filepath) {
  std::ifstream file(filepath, std::ios::binary);
  if (!file) {
    return 0;
  }
  uint64_t hash = 14695981039346656037ULL;
  char buffer[4096];
  while (file.read(buffer, sizeof(buffer))) {
    std::streamsize bytes_read = file.gcount();
    for (std::streamsize i = 0; i < bytes_read; ++i) {
      hash ^= static_cast<uint8_t>(buffer[i]);
      hash *= 1099511628211ULL;
    }
  }
  std::streamsize bytes_read = file.gcount();
  for (std::streamsize i = 0; i < bytes_read; ++i) {
    hash ^= static_cast<uint8_t>(buffer[i]);
    hash *= 1099511628211ULL;
  }
  return hash;
}

bool IsStrategicAction(const std::string& action_str) {
  // Strategic action families where search can provide useful signal.
  if (action_str.rfind("PlaceAgent", 0) == 0 ||
      action_str.rfind("Deploy ", 0) == 0 ||
      action_str.rfind("CombatCommit", 0) == 0 ||
      action_str.rfind("Buy", 0) == 0 ||
      action_str.rfind("AcquireTech", 0) == 0 ||
      action_str.rfind("AcquireTleilaxu", 0) == 0 ||
      action_str.rfind("Shipping", 0) == 0 ||
      action_str.rfind("SignetRing", 0) == 0 ||
      action_str.rfind("tech_", 0) == 0 ||
      action_str.rfind("ResearchBranch", 0) == 0 ||
      action_str.rfind("SelectAgentCard", 0) == 0 ||
      action_str.rfind("SelectGraftPartner", 0) == 0 ||
      action_str.rfind("PlayAgentSolo", 0) == 0 ||
      action_str.rfind("PlayPlotIntrigue", 0) == 0 ||
      action_str.rfind("PlayCombatIntrigue", 0) == 0 ||
      action_str.rfind("IntrigueChoice", 0) == 0 ||
      action_str.rfind("CombatPass", 0) == 0 ||
      action_str.rfind("AgentPass", 0) == 0 ||
      action_str.rfind("Reveal", 0) == 0 ||
      action_str == "PlayKwisatzHaderach" ||
      action_str == "FamilyAtomics" ||
      action_str == "AgentPlayGeneric") {
    return true;
  }
  return false;
}

bool IsStrategicState(const State& state, Player searched_player) {
  if (state.CurrentPlayer() != searched_player) return false;
  std::vector<Action> legal_actions = state.LegalActions();
  if (legal_actions.size() < 2 || legal_actions.size() > 20) return false;

  // Check if ANY action in the set is strategic.
  const dune_imperium::DuneImperiumState& dune_state =
      static_cast<const dune_imperium::DuneImperiumState&>(state);
  for (Action a : legal_actions) {
    std::string action_str = dune_state.ActionToString(state.CurrentPlayer(), a);
    if (IsStrategicAction(action_str)) return true;
  }
  return false;
}

double ComputeEntropy(const ActionsAndProbs& prior) {
  double h = 0.0;
  for (const auto& ap : prior) {
    double p = ap.second;
    if (p > 1e-12) {
      h -= p * std::log(p);
    }
  }
  return h;
}

double CalibrateEta(const ActionsAndProbs& ppo_prior, const std::vector<double>& advantages,
                    double target_kl, double eta_max, double tolerance = 1e-4) {
  double low = 0.0;
  double high = eta_max;
  double best_eta = 0.0;

  auto compute_kl = [&](double eta) {
    double sum_exp = 0.0;
    std::vector<double> probs(ppo_prior.size());
    for (size_t i = 0; i < ppo_prior.size(); ++i) {
      double p = ppo_prior[i].second;
      double adv = advantages[i];
      double val = p * std::exp(eta * adv);
      probs[i] = val;
      sum_exp += val;
    }
    if (sum_exp <= 0.0) return 0.0;
    double kl = 0.0;
    for (size_t i = 0; i < ppo_prior.size(); ++i) {
      double p_teacher = probs[i] / sum_exp;
      if (p_teacher > 1e-12) {
        double p_ppo = ppo_prior[i].second;
        kl += p_teacher * std::log(p_teacher / std::max(p_ppo, 1e-12));
      }
    }
    return kl;
  };

  double kl_at_max = compute_kl(eta_max);
  if (kl_at_max <= target_kl) {
    return eta_max;
  }

  for (int iter = 0; iter < 30; ++iter) {
    double mid = low + (high - low) / 2.0;
    double kl = compute_kl(mid);
    if (std::abs(kl - target_kl) < tolerance) {
      return mid;
    }
    if (kl > target_kl) {
      high = mid;
    } else {
      low = mid;
      best_eta = mid;
    }
  }
  return best_eta;
}

class LabelWriter {
 public:
  LabelWriter(const std::string& dir, int obs_size, int action_dim, int max_simulations,
              float value_scale, float puct_c, float target_teacher_kl,
              int min_visits_per_action, int min_coverage, float blueprint_temperature,
              uint64_t fingerprint, int labels_per_file)
      : dir_(dir), obs_size_(obs_size), action_dim_(action_dim), max_simulations_(max_simulations),
        value_scale_(value_scale), puct_c_(puct_c), target_teacher_kl_(target_teacher_kl),
        min_visits_per_action_(min_visits_per_action), min_coverage_(min_coverage),
        blueprint_temperature_(blueprint_temperature), fingerprint_(fingerprint),
        labels_per_file_(labels_per_file) {
    std::filesystem::create_directories(dir_);
  }

  void WriteLabel(const std::vector<float>& obs, const ActionsAndProbs& ppo_prior,
                  const ActionsAndProbs& teacher_prior, float kl, int num_covered_actions,
                  float eta, bool eta_capped) {
    if (!out_.is_open()) {
      StartNewFile();
    }

    out_.write(reinterpret_cast<const char*>(obs.data()), obs.size() * sizeof(float));
    int32_t num_legal = ppo_prior.size();
    out_.write(reinterpret_cast<const char*>(&num_legal), sizeof(int32_t));
    for (int i = 0; i < num_legal; ++i) {
      int32_t action_id = ppo_prior[i].first;
      float teacher_prob = teacher_prior[i].second;
      float ppo_prob = ppo_prior[i].second;
      out_.write(reinterpret_cast<const char*>(&action_id), sizeof(int32_t));
      out_.write(reinterpret_cast<const char*>(&teacher_prob), sizeof(float));
      out_.write(reinterpret_cast<const char*>(&ppo_prob), sizeof(float));
    }
    out_.write(reinterpret_cast<const char*>(&kl), sizeof(float));
    int32_t covered = num_covered_actions;
    out_.write(reinterpret_cast<const char*>(&covered), sizeof(int32_t));
    out_.write(reinterpret_cast<const char*>(&eta), sizeof(float));
    uint8_t capped = eta_capped ? 1 : 0;
    out_.write(reinterpret_cast<const char*>(&capped), sizeof(uint8_t));
    uint8_t padding[3] = {0, 0, 0};
    out_.write(reinterpret_cast<const char*>(padding), 3);

    labels_in_current_file_++;
    total_labels_written_++;

    if (labels_in_current_file_ >= labels_per_file_) {
      CloseCurrentFile();
    }
  }

  void Close() {
    if (out_.is_open()) {
      CloseCurrentFile();
    }
  }

  int TotalLabelsWritten() const { return total_labels_written_; }

 private:
  void StartNewFile() {
    std::string filename = absl::StrFormat("labels_%d_%lld", getpid(), std::chrono::steady_clock::now().time_since_epoch().count());
    current_bin_path_ = (std::filesystem::path(dir_) / (filename + ".bin")).string();
    current_tmp_path_ = (std::filesystem::path(dir_) / (filename + ".tmp")).string();

    out_.open(current_tmp_path_, std::ios::binary);
    if (!out_) {
      std::cerr << "Failed to open temporary file for writing: " << current_tmp_path_ << "\n";
      std::abort();
    }

    uint32_t magic = 0x4c545344; // "DSTL" in little endian
    uint32_t schema = 1;
    int32_t obs_size = obs_size_;
    int32_t action_dim = action_dim_;
    int32_t max_sim = max_simulations_;
    float val_scale = value_scale_;
    float p_c = puct_c_;
    float t_kl = target_teacher_kl_;
    int32_t min_visits = min_visits_per_action_;
    int32_t min_cov = min_coverage_;
    float b_temp = blueprint_temperature_;
    uint64_t fp = fingerprint_;
    uint32_t reserved = 0;

    out_.write(reinterpret_cast<const char*>(&magic), 4);
    out_.write(reinterpret_cast<const char*>(&schema), 4);
    out_.write(reinterpret_cast<const char*>(&obs_size), 4);
    out_.write(reinterpret_cast<const char*>(&action_dim), 4);
    out_.write(reinterpret_cast<const char*>(&max_sim), 4);
    out_.write(reinterpret_cast<const char*>(&val_scale), 4);
    out_.write(reinterpret_cast<const char*>(&p_c), 4);
    out_.write(reinterpret_cast<const char*>(&t_kl), 4);
    out_.write(reinterpret_cast<const char*>(&min_visits), 4);
    out_.write(reinterpret_cast<const char*>(&min_cov), 4);
    out_.write(reinterpret_cast<const char*>(&b_temp), 4);
    out_.write(reinterpret_cast<const char*>(&fp), 8);
    out_.write(reinterpret_cast<const char*>(&reserved), 4);

    labels_in_current_file_ = 0;
  }

  void CloseCurrentFile() {
    out_.close();
    std::filesystem::rename(current_tmp_path_, current_bin_path_);
    current_tmp_path_ = "";
    current_bin_path_ = "";
    labels_in_current_file_ = 0;
  }

  std::string dir_;
  int obs_size_;
  int action_dim_;
  int max_simulations_;
  float value_scale_;
  float puct_c_;
  float target_teacher_kl_;
  int min_visits_per_action_;
  int min_coverage_;
  float blueprint_temperature_;
  uint64_t fingerprint_;
  int labels_per_file_;

  std::ofstream out_;
  std::string current_tmp_path_;
  std::string current_bin_path_;
  int labels_in_current_file_ = 0;
  int total_labels_written_ = 0;
};

Action SampleBlueprintAction(const ActionsAndProbs& prior, double temp, std::mt19937& rng) {
  if (prior.empty()) return kInvalidAction;
  if (temp <= 0.0) {
    Action best_action = prior[0].first;
    double best_prob = prior[0].second;
    for (const auto& ap : prior) {
      if (ap.second > best_prob) {
        best_prob = ap.second;
        best_action = ap.first;
      }
    }
    return best_action;
  } else {
    std::vector<double> probs;
    probs.reserve(prior.size());
    for (const auto& ap : prior) {
      double p = std::pow(std::max(ap.second, 1e-12), 1.0 / temp);
      probs.push_back(p);
    }
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return prior[dist(rng)].first;
  }
}



} // namespace
} // namespace open_spiel

int main(int argc, char** argv) {
  using namespace open_spiel;
  using namespace open_spiel::dune_imperium;
  absl::ParseCommandLine(argc, argv);


  std::string model_checkpoint = absl::GetFlag(FLAGS_model_checkpoint);
  std::string output_dir = absl::GetFlag(FLAGS_output_dir);
  if (model_checkpoint.empty() || output_dir.empty()) {
    std::cerr << "Error: --model_checkpoint and --output_dir must be specified.\n";
    return 1;
  }

  uint64_t fingerprint = open_spiel::ComputeFileHash(model_checkpoint);
  std::cout << absl::StrFormat("Model fingerprint (FNV-1a 64-bit): 0x%016llx\n", fingerprint);

  torch::manual_seed(absl::GetFlag(FLAGS_seed));

  auto game = open_spiel::LoadGame("dune_imperium");
  int64_t obs_size = game->GetType().provides_information_state_tensor
                         ? game->InformationStateTensorSize()
                         : game->ObservationTensorSize();
  int64_t action_size = game->NumDistinctActions();

  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA);
  }

  auto model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      obs_size, absl::GetFlag(FLAGS_hidden_dim), action_size,
      absl::GetFlag(FLAGS_num_blocks));
  try {
    torch::load(model, model_checkpoint, device);
    std::cout << "Loaded checkpoint model successfully.\n";
  } catch (const c10::Error& e) {
    std::cerr << "Failed to load model checkpoint: " << e.msg() << "\n";
    return 1;
  }
  model->to(device);
  model->eval();

  std::atomic<int> games_started{0};
  std::atomic<int> labels_emitted{0};
  std::atomic<int> total_eta_capped_count{0};
  std::atomic<int> total_search_count{0};
  std::atomic<int> total_search_attempted{0};
  std::atomic<double> total_actual_kl{0.0};
  std::mutex writer_mutex;


  open_spiel::LabelWriter writer(
      output_dir, obs_size, action_size, absl::GetFlag(FLAGS_max_simulations),
      absl::GetFlag(FLAGS_value_scale), absl::GetFlag(FLAGS_puct_c),
      absl::GetFlag(FLAGS_target_teacher_kl), absl::GetFlag(FLAGS_min_visits_per_action),
      absl::GetFlag(FLAGS_min_coverage), absl::GetFlag(FLAGS_blueprint_temperature),
      fingerprint, absl::GetFlag(FLAGS_labels_per_file));

  int target_labels = absl::GetFlag(FLAGS_target_labels);
  int max_games = absl::GetFlag(FLAGS_max_games);
  int num_threads = absl::GetFlag(FLAGS_threads);

  auto worker_fn = [&](int thread_id) {
    auto evaluator = std::make_shared<open_spiel::DuneNNEvaluator>(
        model, device, absl::GetFlag(FLAGS_value_scale));

    while (labels_emitted < target_labels && games_started < max_games) {
      int game_id = games_started.fetch_add(1);
      if (game_id >= max_games) break;
      std::unique_ptr<open_spiel::State> state = game->NewInitialState();
      open_spiel::Player searched_player = game_id % state->NumPlayers();
      int chance_index = 0;
      int decision_index = 0;
      uint64_t master = absl::GetFlag(FLAGS_seed);

      while (!state->IsTerminal()) {
        if (labels_emitted >= target_labels) break;

        if (state->IsChanceNode()) {
          auto outcomes = state->ChanceOutcomes();
          uint64_t chance_seed = dune_seed::DeriveSeed(master, dune_seed::kDomainSearchTeacher, game_id, chance_index, dune_seed::kStreamChance);
          chance_index++;
          auto chance_rng = dune_seed::MakeRng64(chance_seed);
          open_spiel::Action choice = open_spiel::SampleAction(outcomes, absl::Uniform(chance_rng, 0.0, 1.0)).first;
          state->ApplyAction(choice);
          continue;
        }

        open_spiel::Player cur_player = state->CurrentPlayer();
        open_spiel::ActionsAndProbs ppo_prior = evaluator->Prior(*state);

        uint64_t blueprint_seed = dune_seed::DeriveSeed(master, dune_seed::kDomainSearchTeacher, game_id, decision_index, dune_seed::kStreamBlueprint);
        auto blueprint_rng = dune_seed::MakeRng32(blueprint_seed);
        open_spiel::Action blueprint_action = open_spiel::SampleBlueprintAction(ppo_prior, absl::GetFlag(FLAGS_blueprint_temperature), blueprint_rng);

        bool should_search = (cur_player == searched_player && open_spiel::IsStrategicState(*state, searched_player));

        if (should_search) {
          double entropy = open_spiel::ComputeEntropy(ppo_prior);
          double max_entropy = std::log(state->LegalActions().size());
          double entropy_ratio = max_entropy > 0.0 ? (entropy / max_entropy) : 0.0;
          double search_prob = absl::GetFlag(FLAGS_search_fraction);
          if (entropy_ratio < absl::GetFlag(FLAGS_min_entropy_ratio)) {
            search_prob = absl::GetFlag(FLAGS_search_fraction) * absl::GetFlag(FLAGS_uniform_ratio);
          }

          uint64_t search_gate_seed = dune_seed::DeriveSeed(master, dune_seed::kDomainSearchTeacher, game_id, decision_index, dune_seed::kStreamSearchGate);
          auto search_gate_rng = dune_seed::MakeRng64(search_gate_seed);
          if (absl::Uniform(search_gate_rng, 0.0, 1.0) < search_prob) {
            total_search_attempted++;
            uint64_t mcts_seed = dune_seed::DeriveSeed(master, dune_seed::kDomainSearchTeacher, game_id, decision_index, dune_seed::kStreamMCTS);
            open_spiel::DunePUCTISMCTSBot bot(
                mcts_seed, evaluator, absl::GetFlag(FLAGS_puct_c), absl::GetFlag(FLAGS_max_simulations),
                -1, 1.0, 0.0, 0.3, 1.0, true,
                open_spiel::DuneISMCTSFinalPolicyType::kNormalizedVisitCount,
                true, absl::GetFlag(FLAGS_search_opponent_temperature));
            bot.RunSearch(*state);
            open_spiel::SearchDiagnostics diag = bot.GetRootDiagnostics(*state, absl::GetFlag(FLAGS_min_visits_per_action));

            int required_coverage = std::min(absl::GetFlag(FLAGS_min_coverage), static_cast<int>(diag.actions.size()));
            bool has_coverage = (diag.num_covered_actions >= required_coverage);
            bool has_mass = (diag.covered_prior_mass >= absl::GetFlag(FLAGS_min_prior_mass));

            if (has_coverage && has_mass) {
              std::vector<double> Q_teacher(diag.actions.size());
              for (size_t i = 0; i < diag.actions.size(); ++i) {
                if (diag.visit_counts[i] >= absl::GetFlag(FLAGS_min_visits_per_action)) {
                  Q_teacher[i] = diag.q_values[i];
                } else {
                  Q_teacher[i] = diag.root_value;
                }
              }

              double V_baseline = 0.0;
              for (size_t i = 0; i < diag.actions.size(); ++i) {
                V_baseline += diag.priors[i] * Q_teacher[i];
              }

              std::vector<double> advantages(diag.actions.size());
              for (size_t i = 0; i < diag.actions.size(); ++i) {
                advantages[i] = Q_teacher[i] - V_baseline;
              }

              double target_kl = absl::GetFlag(FLAGS_target_teacher_kl);
              double eta_max = absl::GetFlag(FLAGS_eta_max);
              double eta = open_spiel::CalibrateEta(ppo_prior, advantages, target_kl, eta_max);
              bool eta_capped = (std::abs(eta - eta_max) < 1e-4);

              double sum_exp = 0.0;
              open_spiel::ActionsAndProbs teacher_prior;
              teacher_prior.reserve(diag.actions.size());
              for (size_t i = 0; i < diag.actions.size(); ++i) {
                double val = diag.priors[i] * std::exp(eta * advantages[i]);
                teacher_prior.push_back({diag.actions[i], val});
                sum_exp += val;
              }
              for (auto& ap : teacher_prior) {
                ap.second /= sum_exp;
              }

              double kl = 0.0;
              for (size_t i = 0; i < teacher_prior.size(); ++i) {
                double p_t = teacher_prior[i].second;
                if (p_t > 1e-12) {
                  kl += p_t * std::log(p_t / std::max(diag.priors[i], 1e-12));
                }
              }

              std::vector<float> obs;
              if (game->GetType().provides_information_state_tensor) {
                obs = state->InformationStateTensor(cur_player);
              } else {
                obs.resize(obs_size);
                state->ObservationTensor(cur_player, absl::MakeSpan(obs));
              }

              {
                std::lock_guard<std::mutex> lock(writer_mutex);
                if (labels_emitted < target_labels) {
                  writer.WriteLabel(obs, ppo_prior, teacher_prior, kl, diag.num_covered_actions, eta, eta_capped);
                  labels_emitted++;
                  if (eta_capped) total_eta_capped_count++;
                  total_search_count++;
                  double prev = total_actual_kl.load(std::memory_order_relaxed);
                  while (!total_actual_kl.compare_exchange_weak(
                      prev, prev + kl,
                      std::memory_order_relaxed, std::memory_order_relaxed)) {}
                }
              }
            }
          }
        }

        decision_index++;
        state->ApplyAction(blueprint_action);
      }
    }
  };

  std::vector<std::thread> workers;
  for (int i = 0; i < num_threads; ++i) {
    workers.emplace_back(worker_fn, i);
  }

  for (auto& t : workers) {
    t.join();
  }

  writer.Close();

  std::cout << "\n=== Generator Finished ===\n";
  std::cout << absl::StrFormat("Total games played: %d\n", games_started.load());
  std::cout << absl::StrFormat("Total labels generated: %d\n", writer.TotalLabelsWritten());
  int accepted = total_search_count.load();
  int attempted = total_search_attempted.load();
  if (attempted > 0) {
    double yield = (100.0 * accepted) / attempted;
    std::cout << absl::StrFormat("Label yield: %.2f%% (%d accepted / %d searched)\n",
                                 yield, accepted, attempted);
  }
  if (accepted > 0) {
    double mean_kl = total_actual_kl.load(std::memory_order_relaxed) / accepted;
    double cap_rate = (100.0 * total_eta_capped_count.load()) / accepted;
    std::cout << absl::StrFormat("Mean actual teacher KL: %.4f nats\n", mean_kl);
    std::cout << absl::StrFormat("Eta cap hit rate: %.2f%% (%d/%d accepted)\n",
                                 cap_rate, total_eta_capped_count.load(), accepted);
  }
  if (games_started.load() > 0) {
    double labels_per_game = static_cast<double>(writer.TotalLabelsWritten()) / games_started.load();
    std::cout << absl::StrFormat("Labels per game: %.1f\n", labels_per_game);
  }

  return 0;
}
