#ifndef OPEN_SPIEL_EXAMPLES_DUNE_PUCT_IS_MCTS_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_PUCT_IS_MCTS_H_

#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/algorithms/mcts.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"

namespace open_spiel {

struct DuneChildInfo {
  int visits = 0;
  double return_sum = 0.0;
  double prior = 0.0;
  double value() const { return visits > 0 ? return_sum / visits : 0.0; }
};

struct DuneISMCTSNode {
  absl::flat_hash_map<Action, DuneChildInfo> child_info;
  int total_visits = -1;
  bool priors_initialized = false;
};

enum class DuneISMCTSFinalPolicyType {
  kNormalizedVisitCount,
  kMaxVisitCount,
  kMaxValue,
};

struct TestBotAccessor;

class DunePUCTISMCTSBot : public Bot {
  friend struct TestBotAccessor;
 public:
  DunePUCTISMCTSBot(int seed, std::shared_ptr<algorithms::Evaluator> evaluator,
                    double puct_c, int max_simulations,
                    int max_world_samples = -1,
                    double temperature = 1.0,
                    double dirichlet_epsilon = 0.0,
                    double dirichlet_alpha = 0.3,
                    double value_scale = 4.0,
                    bool use_observation_string = true,
                    bool allow_inconsistent_action_sets = true,
                    DuneISMCTSFinalPolicyType final_policy_type = DuneISMCTSFinalPolicyType::kNormalizedVisitCount,
                    bool use_opponent_model = false,
                    double opponent_temperature = 0.0,
                    bool verbose_diagnostics = false);

  Action Step(const State& state) override;
  bool ProvidesPolicy() override { return true; }
  ActionsAndProbs GetPolicy(const State& state) override;
  std::pair<ActionsAndProbs, Action> StepWithPolicy(const State& state) override;

  ActionsAndProbs RunSearch(const State& state);

  // Bot maintains no history, so these are empty.
  void Restart() override {}
  void RestartAt(const State& state) override {}

 private:
  void Reset();
  double RandomNumber();
  std::pair<Player, std::string> GetStateKey(const State& state) const;
  std::unique_ptr<State> SampleRootState(const State& state);
  std::unique_ptr<State> ResampleFromInfostate(const State& state);
  DuneISMCTSNode* LookupOrCreateNode(const State& state);
  DuneISMCTSNode* CreateNewNode(const State& state);
  DuneISMCTSNode* LookupNode(const State& state);

  // Core search procedures
  std::vector<double> RunSimulation(State* state, int depth = 0);
  Action SelectActionTreePolicy(DuneISMCTSNode* node, const std::vector<Action>& legal_actions);
  Action SelectActionPUCT(DuneISMCTSNode* node);
  void InitializePriors(DuneISMCTSNode* node, const State& state);
  ActionsAndProbs FilterAndNormalizePriors(DuneISMCTSNode* node, const std::vector<Action>& legal_actions) const;
  ActionsAndProbs GetFinalPolicy(const State& state, DuneISMCTSNode* node) const;

  std::mt19937 rng_;
  std::shared_ptr<algorithms::Evaluator> evaluator_;
  double puct_c_;
  int max_simulations_;
  int max_world_samples_;
  double temperature_;
  double dirichlet_epsilon_;
  double dirichlet_alpha_;
  double value_scale_;
  bool use_observation_string_;
  bool allow_inconsistent_action_sets_;
  DuneISMCTSFinalPolicyType final_policy_type_;
  bool use_opponent_model_;
  double opponent_temperature_;
  bool verbose_diagnostics_;

  Player searching_player_ = kInvalidPlayer;
  int max_depth_this_search_ = 0;
  double sum_depth_this_search_ = 0.0;
  int num_sims_this_search_ = 0;
  int total_lookups_ = 0;
  int reused_lookups_ = 0;
  int search_count_ = 0;

  absl::flat_hash_map<std::pair<Player, std::string>, DuneISMCTSNode*> nodes_;
  std::vector<std::unique_ptr<DuneISMCTSNode>> node_pool_;
  std::vector<std::unique_ptr<State>> root_samples_;
  DuneISMCTSNode* root_node_;
};

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_PUCT_IS_MCTS_H_
