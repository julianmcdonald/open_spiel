#ifndef OPEN_SPIEL_EXAMPLES_DUNE_FROZEN_POLICY_DIAG_EVALUATORS_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_FROZEN_POLICY_DIAG_EVALUATORS_H_

#include "dune_frozen_policy_diag_types.h"

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <torch/torch.h>
#include "dune_network.h"
#include "dune_semantic_action_scorer.h"
#include "dune_sha256.h"
#endif

namespace open_spiel {
namespace dune_diagnostics {

// ===========================================================================
// Evaluator Interface & Scripted Controls
// ===========================================================================

class IPolicyDiagnosticsEvaluator {
 public:
  virtual ~IPolicyDiagnosticsEvaluator() = default;
  virtual ActionsAndProbs GetPolicy(const State& state, Player player) = 0;
  virtual Action SelectAction(const State& state, Player player, bool greedy,
                              float temperature, std::mt19937_64& rng) = 0;
  virtual std::vector<float> GetRawLogits(const State& state, Player player) = 0;
  virtual bool IsNeural() const = 0;
  virtual std::string ModelName() const = 0;
  virtual std::string ModelSha256() const = 0;
  virtual ModelInputSnapshot CaptureSnapshot(const State& state, Player player) {
    ModelInputSnapshot snap;
    snap.state_string = state.ToString();
    snap.legal_actions = GetPlayerLegalActions(state, player);
    snap.observation_tensor = state.InformationStateTensor(player);
    snap.policy = GetPolicy(state, player);
    return snap;
  }
};

// Positive control: Observes P2's visible solari.
// If P2 has >= 8 solari, P1 aggressively takes Swordmaster.
// If P2 has < 8 solari, P1 takes an alternative space.
class ScriptedPositiveControlEvaluator : public IPolicyDiagnosticsEvaluator {
 public:
  explicit ScriptedPositiveControlEvaluator(int p_holder, int p_competitor)
      : p_holder_(p_holder), p_competitor_(p_competitor) {}

  bool IsNeural() const override { return false; }
  std::string ModelName() const override { return "ScriptedPositiveControl"; }
  std::string ModelSha256() const override { return "positive_control_sha256_synthetic"; }

  std::vector<float> GetRawLogits(const State& state, Player player) override {
    auto legal = GetPlayerLegalActions(state, player);
    std::vector<float> logits(3000, -100.0f);
    for (Action a : legal) logits[a] = 0.0f;
    return logits;
  }

  ActionsAndProbs GetPolicy(const State& state, Player player) override {
    auto legal = GetPlayerLegalActions(state, player);
    ActionsAndProbs policy;
    if (legal.empty()) return policy;
    double p = 1.0 / legal.size();
    for (Action a : legal) policy.push_back({a, p});
    return policy;
  }

  Action SelectAction(const State& state, Player player, bool /*greedy*/,
                      float /*temperature*/, std::mt19937_64& /*rng*/) override {
    const auto* dune = dynamic_cast<const DuneImperiumState*>(&state);
    SPIEL_CHECK_TRUE(dune != nullptr);
    auto legal = state.LegalActions();
    SPIEL_CHECK_FALSE(legal.empty());

    if (player == p_competitor_) {
      int holder_solari = dune->GetPlayerSolari(p_holder_);
      bool threat = (holder_solari >= 8);

      // Card selection phase
      if (dune_imperium::HasCardSelectionAction(legal)) {
        if (threat) {
          // Select card with Landsraad access to enable Swordmaster
          for (Action a : legal) {
            if (a >= kActionSelectAgentCard0 && a < kActionSelectAgentCard0 + 256) {
              int cid = a - kActionSelectAgentCard0;
              const auto* card = FindImperiumCardById(cid);
              if (card && (card->access_mask & dune_imperium::kAccessLandsraad)) {
                return a;
              }
            }
          }
        } else {
          // Non-threat: pick a card that does NOT access Landsraad if possible
          for (Action a : legal) {
            if (a >= kActionSelectAgentCard0 && a < kActionSelectAgentCard0 + 256) {
              int cid = a - kActionSelectAgentCard0;
              const auto* card = FindImperiumCardById(cid);
              if (card && !(card->access_mask & dune_imperium::kAccessLandsraad)) {
                return a;
              }
            }
          }
        }
        return legal[0];
      }

      // Solo play
      if (ContainsAction(legal, kActionPlayAgentSolo)) {
        return kActionPlayAgentSolo;
      }

      // Graft partner selection
      for (Action a : legal) {
        if (a >= kActionSelectGraftPartner0 && a < kActionSelectGraftPartner0 + 256) {
          return a;
        }
      }

      // Board space placement
      if (threat && ContainsAction(legal, kActionAgentSpaceSwordmaster)) {
        return kActionAgentSpaceSwordmaster;
      }

      // Otherwise pick an alternative space or legal action
      for (Action a : legal) {
        if (IsAgentSpaceAction(a) && a != kActionAgentSpaceSwordmaster) {
          return a;
        }
      }
      return legal[0];
    }

    // Default player behavior
    return dune_imperium::DefaultProgressionAction(legal);
  }

 private:
  int p_holder_;
  int p_competitor_;
};

// ===========================================================================
// Scripted Nonreactive Control Evaluator
// ===========================================================================

class ScriptedNonreactiveControlEvaluator : public IPolicyDiagnosticsEvaluator {
 private:
  int p_holder_ = 0;
  int p_competitor_ = 1;

 public:
  ScriptedNonreactiveControlEvaluator(int p_holder, int p_competitor)
      : p_holder_(p_holder), p_competitor_(p_competitor) {}

  bool IsNeural() const override { return false; }
  std::string ModelName() const override { return "ScriptedNonreactiveControl"; }
  std::string ModelSha256() const override { return "nonreactive_control_sha256_synthetic"; }

  std::vector<float> GetRawLogits(const State& state, Player player) override {
    auto legal = GetPlayerLegalActions(state, player);
    std::vector<float> logits(3000, -100.0f);
    for (Action a : legal) logits[a] = 0.0f;
    return logits;
  }

  ActionsAndProbs GetPolicy(const State& state, Player player) override {
    auto legal = GetPlayerLegalActions(state, player);
    ActionsAndProbs policy;
    if (legal.empty()) return policy;
    double p = 1.0 / legal.size();
    for (Action a : legal) policy.push_back({a, p});
    return policy;
  }

  Action SelectAction(const State& state, Player player, bool /*greedy*/,
                      float /*temperature*/, std::mt19937_64& /*rng*/) override {
    const auto* dune = dynamic_cast<const DuneImperiumState*>(&state);
    SPIEL_CHECK_TRUE(dune != nullptr);
    auto legal = state.LegalActions();
    SPIEL_CHECK_FALSE(legal.empty());

    if (player == p_competitor_) {
      // Fixed priority regardless of P2's solari
      if (dune_imperium::HasCardSelectionAction(legal)) {
        for (Action a : legal) {
          if (a >= kActionSelectAgentCard0 && a < kActionSelectAgentCard0 + 256) return a;
        }
      }
      if (ContainsAction(legal, kActionPlayAgentSolo)) return kActionPlayAgentSolo;
      for (Action a : legal) {
        if (a >= kActionSelectGraftPartner0 && a < kActionSelectGraftPartner0 + 256) return a;
      }
      // Fixed board space preference: High Council > Mentat > Swordmaster > Arrakeen
      if (ContainsAction(legal, kActionAgentSpaceHighCouncil)) return kActionAgentSpaceHighCouncil;
      if (ContainsAction(legal, kActionAgentSpaceMentat)) return kActionAgentSpaceMentat;
      if (ContainsAction(legal, kActionAgentSpaceSwordmaster)) return kActionAgentSpaceSwordmaster;
      if (ContainsAction(legal, kActionAgentSpaceArrakeen)) return kActionAgentSpaceArrakeen;
      for (Action a : legal) {
        if (IsAgentSpaceAction(a)) return a;
      }
      if (ContainsAction(legal, kActionReveal)) return kActionReveal;
      if (ContainsAction(legal, kActionEndTurn)) return kActionEndTurn;
      return legal[0];
    }

    if (ContainsAction(legal, kActionEndTurn)) return kActionEndTurn;
    return dune_imperium::TraceSimulationAction(legal);
  }
};

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
class NeuralPolicyDiagnosticsEvaluator : public IPolicyDiagnosticsEvaluator {
 public:
  NeuralPolicyDiagnosticsEvaluator(const std::string& checkpoint_path,
                                   torch::Device device,
                                   float logit_cap = 10.0f)
      : path_(checkpoint_path), device_(device), logit_cap_(logit_cap) {
    std::cout << "Loading neural checkpoint for diagnostics: " << checkpoint_path << std::endl;
    sha256_ = ComputeFileSHA256(checkpoint_path);
    std::cout << "Checkpoint SHA256: " << sha256_ << std::endl;

    // Detect architecture and schema
    ReadMetadata();

    model_ = std::make_shared<SharedDunePolicyValueNetImpl>(
        obs_dim_, hidden_dim_, kActionDim, num_blocks_,
        /*use_nonlinear=*/false, /*with_aux_heads=*/false,
        /*head_init_seed=*/0, with_semantic_scorer_);
    model_->deck_pool_semantics_ = deck_pool_semantics_;

    torch::serialize::InputArchive archive;
    archive.load_from(checkpoint_path, device_);
    model_->load(archive);
    model_->to(device_);
    model_->eval();

    evaluator_ = std::make_shared<DeterministicEvaluator>(
        model_, device_, &eval_mutex_, nullptr, nullptr, /*rollout_amp=*/false);
  }

  bool IsNeural() const override { return true; }
  std::string ModelName() const override { return path_; }
  std::string ModelSha256() const override { return sha256_; }

  int ObservationDim() const { return obs_dim_; }
  int HiddenDim() const { return hidden_dim_; }
  int NumBlocks() const { return num_blocks_; }
  bool WithSemanticScorer() const { return with_semantic_scorer_; }
  dune_imperium::MarketAppendixMode MarketMode() const { return market_mode_; }
  dune_imperium::DeckPoolSemantics DeckPoolSemantics() const { return deck_pool_semantics_; }
  bool HasSemanticScorer() const { return evaluator_ != nullptr && evaluator_->HasSemanticScorer(); }
  std::string SemanticDescriptorSchema() const {
    return evaluator_ != nullptr ? evaluator_->SemanticDescriptorSchema() : dune_semantic::kDescriptorSchemaVersionV3;
  }

  ModelInputSnapshot CaptureSnapshot(const State& state, Player player) override {
    ModelInputSnapshot snap;
    snap.state_string = state.ToString();
    snap.legal_actions = GetPlayerLegalActions(state, player);
    const auto* dune = dynamic_cast<const DuneImperiumState*>(&state);
    if (dune != nullptr) {
      snap.observation_tensor = dune->InformationStateTensorWithAppendix(
          player, market_mode_, DeckPoolSemantics());
      if (evaluator_ != nullptr && evaluator_->HasSemanticScorer() && !snap.legal_actions.empty()) {
        dune_semantic::CandidateActionData cand_data;
        dune_semantic::ExtractCandidateDescriptors(
            *dune, snap.legal_actions, &cand_data, evaluator_->SemanticDescriptorSchema());
        snap.action_features = cand_data.features;
        snap.action_card_ids = cand_data.card_ids;
        snap.action_space_ids = cand_data.space_ids;
        snap.action_supported = cand_data.supported;
      }
    } else {
      snap.observation_tensor = state.InformationStateTensor(player);
    }
    snap.policy = GetPolicy(state, player);
    return snap;
  }

  std::vector<float> GetRawLogits(const State& state, Player player) override {
    const auto* dune = dynamic_cast<const DuneImperiumState*>(&state);
    std::vector<float> obs;
    if (dune != nullptr) {
      obs = dune->InformationStateTensorWithAppendix(player, market_mode_, DeckPoolSemantics());
    } else {
      obs = state.InformationStateTensor(player);
    }
    std::vector<Action> legal = GetPlayerLegalActions(state, player);

    std::unique_ptr<dune_semantic::CandidateActionData> cand_data;
    if (evaluator_->HasSemanticScorer() && !legal.empty()) {
      cand_data = std::make_unique<dune_semantic::CandidateActionData>();
      dune_semantic::ExtractCandidateDescriptors(
          *dune, legal, cand_data.get(), evaluator_->SemanticDescriptorSchema());
    }

    auto res = evaluator_->EvaluateWithActions(obs, cand_data.get());
    return res.logits;
  }

  ActionsAndProbs GetPolicy(const State& state, Player player) override {
    auto legal = GetPlayerLegalActions(state, player);
    if (legal.empty()) return {};

    std::vector<float> logits = GetRawLogits(state, player);
    CenterAndCapLegalLogitsWithStats(logits, legal, logit_cap_);

    double max_logit = -1e30;
    for (Action a : legal) {
      if (a >= 0 && static_cast<size_t>(a) < logits.size()) {
        max_logit = std::max(max_logit, static_cast<double>(logits[a]));
      }
    }
    double sum = 0.0;
    std::vector<double> probs(legal.size());
    for (size_t i = 0; i < legal.size(); ++i) {
      Action a = legal[i];
      if (a >= 0 && static_cast<size_t>(a) < logits.size()) {
        probs[i] = std::exp(static_cast<double>(logits[a]) - max_logit);
        sum += probs[i];
      }
    }
    ActionsAndProbs policy;
    policy.reserve(legal.size());
    for (size_t i = 0; i < legal.size(); ++i) {
      double p = (sum > 0.0) ? (probs[i] / sum) : (1.0 / legal.size());
      policy.push_back({legal[i], p});
    }
    return policy;
  }

  Action SelectAction(const State& state, Player player, bool greedy,
                      float temperature, std::mt19937_64& rng) override {
    auto legal = state.LegalActions();
    if (legal.empty()) return open_spiel::kInvalidAction;
    if (legal.size() == 1) return legal[0];

    std::vector<float> logits = GetRawLogits(state, player);
    CenterAndCapLegalLogitsWithStats(logits, legal, logit_cap_);

    dune_eval::SelectionPolicy sp{greedy, temperature};
    return dune_eval::SelectActionFromLogits(logits, legal, sp, rng);
  }

 private:
  void ReadMetadata() {
    std::filesystem::path p(path_);
    std::filesystem::path p_json = p;
    p_json.replace_extension(".json");
    if (std::filesystem::exists(p_json)) {
      std::ifstream f(p_json);
      if (f) {
        std::string content((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
        auto val_opt = json::FromString(content);
        if (val_opt.has_value() && val_opt->IsObject()) {
          const auto& obj = val_opt->GetObject();
          auto it_od = obj.find("observation_dim");
          if (it_od != obj.end() && it_od->second.IsInt()) {
            obs_dim_ = it_od->second.GetInt();
          }
          auto it_hd = obj.find("hidden_dim");
          if (it_hd != obj.end() && it_hd->second.IsInt()) {
            hidden_dim_ = it_hd->second.GetInt();
          }
          auto it_nb = obj.find("num_blocks");
          if (it_nb != obj.end() && it_nb->second.IsInt()) {
            num_blocks_ = it_nb->second.GetInt();
          }
          auto it_ess = obj.find("enable_semantic_scorer");
          if (it_ess != obj.end()) {
            with_semantic_scorer_ = (it_ess->second.IsBool() && it_ess->second.GetBool()) ||
                                   (it_ess->second.IsString() && it_ess->second.GetString() == "true");
          }
          auto it_mm = obj.find("market_appendix_mode");
          if (it_mm != obj.end() && it_mm->second.IsString()) {
            market_mode_str_ = it_mm->second.GetString();
          }
          auto it_sem = obj.find("deck_pool_semantics");
          if (it_sem != obj.end() && it_sem->second.IsString()) {
            deck_pool_semantics_ = dune_imperium::ParseDeckPoolSemanticsOrDie(it_sem->second.GetString());
          }
        }
      }
    }

    if (obs_dim_ == dune_imperium::kFullPublicInformationStateSize) {
      market_mode_ = dune_imperium::MarketAppendixMode::kFullPublicInformationV3;
      with_semantic_scorer_ = true;
    } else if (obs_dim_ == dune_imperium::kOrderedCardSlotsInformationStateSize) {
      market_mode_ = dune_imperium::MarketAppendixMode::kOrderedCardSlotsV2;
    } else {
      market_mode_ = dune_imperium::MarketAppendixMode::kNone;
    }
  }

  std::string path_;
  std::string sha256_;
  torch::Device device_;
  float logit_cap_ = 10.0f;
  int obs_dim_ = 9182;
  int hidden_dim_ = 2048;
  int num_blocks_ = 8;
  static constexpr int kActionDim = 2391;
  bool with_semantic_scorer_ = true;
  std::string market_mode_str_ = "full_public_information_v3";
  dune_imperium::MarketAppendixMode market_mode_ =
      dune_imperium::MarketAppendixMode::kFullPublicInformationV3;
  dune_imperium::DeckPoolSemantics deck_pool_semantics_ =
      dune_imperium::DeckPoolSemantics::kLegacy;

  std::mutex eval_mutex_;
  std::shared_ptr<SharedDunePolicyValueNetImpl> model_;
  std::shared_ptr<DeterministicEvaluator> evaluator_;
};
#endif

// ===========================================================================
// Windfall-Timing Controller & Decision Tracking
// ===========================================================================

class WindfallTimingControlledEvaluator : public IPolicyDiagnosticsEvaluator {
 public:
  WindfallTimingControlledEvaluator(IPolicyDiagnosticsEvaluator* base_evaluator,
                                   int designated_holder = 0)
      : base_evaluator_(base_evaluator), designated_holder_(designated_holder) {
    SPIEL_CHECK_TRUE(base_evaluator_ != nullptr);
  }

  void SetHolder(int holder) { designated_holder_ = holder; }
  void SetContext(int candidate_id, int replicate_id) {
    candidate_id_ = candidate_id;
    replicate_id_ = replicate_id;
    windfall_played_this_run_ = false;
    resource_followup_recorded_ = false;
    last_resource_action_ = open_spiel::kInvalidAction;
    last_resource_action_name_.clear();
  }

  void ResetTurnStep() { turn_step_ = 0; }

  const std::vector<WindfallDecisionRecord>& DecisionRecords() const {
    return decision_records_;
  }

  void ClearDecisionRecords() { decision_records_.clear(); }

  bool IsNeural() const override { return base_evaluator_->IsNeural(); }
  std::string ModelName() const override {
    return base_evaluator_->ModelName() + " [WindfallControlled]";
  }
  std::string ModelSha256() const override { return base_evaluator_->ModelSha256(); }

  std::vector<float> GetRawLogits(const State& state, Player player) override {
    return base_evaluator_->GetRawLogits(state, player);
  }

  ActionsAndProbs GetPolicy(const State& state, Player player) override {
    return base_evaluator_->GetPolicy(state, player);
  }

  ModelInputSnapshot CaptureSnapshot(const State& state, Player player) override {
    return base_evaluator_->CaptureSnapshot(state, player);
  }

  void RecordResourceFollowUp(Action action) {
    if (windfall_played_this_run_ && !resource_followup_recorded_) {
      if (dune_imperium::IsAgentSpaceAction(action) ||
          (action >= dune_imperium::kActionBuyImperiumRow0 && action <= dune_imperium::kActionBuyImperiumRow0 + 10) ||
          action == dune_imperium::kActionBuyReserveArrakisLiaison ||
          action == dune_imperium::kActionBuyReserveTheSpiceMustFlow ||
          (action >= dune_imperium::kActionTechAcquire0 && action <= dune_imperium::kActionTechAcquire0 + 6)) {
        resource_followup_recorded_ = true;
        last_resource_action_ = action;
        last_resource_action_name_ = DiagnosticActionToString(action);
        if (!decision_records_.empty()) {
          decision_records_.back().resource_use_followed = last_resource_action_name_;
        }
      }
    }
  }

  Action SelectAction(const State& state, Player player, bool greedy,
                      float temperature, std::mt19937_64& rng) override {
    turn_step_++;

    // 1. Controller applies ONLY to designated holder
    if (player != designated_holder_) {
      return base_evaluator_->SelectAction(state, player, greedy, temperature, rng);
    }

    const auto* dune = dynamic_cast<const DuneImperiumState*>(&state);
    SPIEL_CHECK_TRUE(dune != nullptr);

    auto legal = state.LegalActions();
    if (legal.empty()) return open_spiel::kInvalidAction;
    if (legal.size() == 1) {
      RecordResourceFollowUp(legal[0]);
      return legal[0];
    }

    // 2. Controller applies ONLY while player possesses Windfall
    const auto& intrigues = dune->GetPlayerIntrigues(player);
    bool possesses_windfall =
        (std::find(intrigues.begin(), intrigues.end(), kIntrigueWindfall) != intrigues.end());

    // Check if Windfall is currently legal to play
    bool windfall_legal = ContainsAction(legal, kActionPlayWindfall);
    if (!possesses_windfall || !windfall_legal) {
      Action act = base_evaluator_->SelectAction(state, player, greedy, temperature, rng);
      RecordResourceFollowUp(act);
      return act;
    }

    // At this point: player possesses Windfall AND kActionPlayWindfall is currently legal!
    WindfallDecisionRecord rec;
    rec.candidate_id = candidate_id_;
    rec.replicate_id = replicate_id_;
    rec.round = dune->GetCurrentRound();
    rec.turn_step = turn_step_;
    rec.player = player;
    rec.actual_solari = dune->GetPlayerSolari(player);
    rec.windfall_available = true;

    // 3. Secrets exception:
    // "For this prototype, conservatively treat holding four or more intrigue cards as exposure.
    //  In that condition, preserve the baseline policy’s Windfall decision rather than forcing retention."
    if (intrigues.size() >= 4) {
      rec.secrets_exception = true;
      Action act = base_evaluator_->SelectAction(state, player, greedy, temperature, rng);
      rec.action_taken = act;
      rec.action_taken_name = DiagnosticActionToString(act);
      rec.windfall_played = (act == kActionPlayWindfall);
      rec.windfall_reason = "Secrets exception: holder possesses >= 4 intrigue cards (exposure risk); baseline decision preserved";
      if (rec.windfall_played) {
        rec.solari_after = rec.actual_solari + 2;
        windfall_played_this_run_ = true;
      } else {
        rec.solari_after = rec.actual_solari;
      }
      decision_records_.push_back(rec);
      RecordResourceFollowUp(act);
      return act;
    }

    // 4. Terminal Value Exception:
    // "Also preserve any terminal value of Windfall’s resources under the engine’s rules.
    //  The controller must not accidentally forfeit usable resources at game end because no purchase remains."
    bool potential_endgame = (dune->GetCurrentRound() >= 10);
    for (int p = 0; p < kNumPlayers; ++p) {
      if (dune->GetPlayerVp(p) >= 10) {
        potential_endgame = true;
        break;
      }
    }

    bool has_economic_positioning =
        (std::find(intrigues.begin(), intrigues.end(), kIntrigueEconomicPositioning) != intrigues.end());
    bool economic_positioning_threshold =
        (has_economic_positioning && (rec.actual_solari == 8 || rec.actual_solari == 9));

    if (economic_positioning_threshold ||
        (potential_endgame && (dune->phase() == GamePhase::kRevealTurns ||
                               dune->GetPlayerAgentsRemainingForTesting(player) == 0))) {
      rec.terminal_value_exception = true;
      rec.plan_requires_windfall = true;
      rec.windfall_reason = economic_positioning_threshold
          ? "Terminal value: Economic Positioning threshold (solari reaches >= 10 for +1 VP)"
          : "Terminal value: Endgame round potential; playing Windfall preserves solari tiebreaker";
      rec.action_taken = kActionPlayWindfall;
      rec.action_taken_name = "PlayPlotIntrigue(Windfall)";
      rec.windfall_played = true;
      rec.solari_after = rec.actual_solari + 2;
      windfall_played_this_run_ = true;
      decision_records_.push_back(rec);
      return kActionPlayWindfall;
    }

    // 5. Speculative Proposal in temporary clone with Windfall resolved:
    // "Use the native policy to propose an immediate action plan from a temporary clone in which Windfall has been legally resolved."
    try {
      auto temp_clone = state.Clone();
      temp_clone->ApplyAction(kActionPlayWindfall);

      // Dedicated proposal RNG stream derived from player's context to avoid polluting real stream
      uint64_t prop_seed = dune_seed::DeriveSeed(20260915, 0x00B0, turn_step_, player, dune_seed::kStreamChance);
      std::mt19937_64 prop_chance_rng(prop_seed);
      std::mt19937_64 prop_policy_rng(prop_seed ^ 0xFEEDFACEULL);

      AdvanceStatus st = AdvanceThroughChanceAndAcks(temp_clone.get(), prop_chance_rng);
      if (st != AdvanceStatus::kSuccessDecision && st != AdvanceStatus::kSuccessTerminal) {
        throw std::runtime_error("Proposal clone advance failed: " + std::string(AdvanceStatusToString(st)));
      }

      if (temp_clone->IsTerminal() || temp_clone->CurrentPlayer() != player) {
        throw std::runtime_error("Proposal clone unexpected player transition");
      }

      auto prop_legal = temp_clone->LegalActions();
      if (prop_legal.empty()) {
        throw std::runtime_error("Proposal clone legal actions empty");
      }

      // Query policy in temp_clone for proposal
      Action a1 = base_evaluator_->SelectAction(*temp_clone, player, /*greedy=*/false, temperature, prop_policy_rng);
      rec.proposed_action_chain.push_back(a1);
      rec.proposed_action_names.push_back(DiagnosticActionToString(a1));

      // Case A: a1 is agent card selection
      if (a1 >= kActionSelectAgentCard0 && a1 < kActionSelectAgentCard0 + 256) {
        temp_clone->ApplyAction(a1);
        AdvanceThroughChanceAndAcks(temp_clone.get(), prop_chance_rng);

        auto next_legal = temp_clone->LegalActions();
        if (temp_clone->CurrentPlayer() == player && !next_legal.empty()) {
          if (ContainsAction(next_legal, kActionPlayAgentSolo)) {
            temp_clone->ApplyAction(kActionPlayAgentSolo);
            rec.proposed_action_chain.push_back(kActionPlayAgentSolo);
            rec.proposed_action_names.push_back("PlayAgentSolo");
            AdvanceThroughChanceAndAcks(temp_clone.get(), prop_chance_rng);
            next_legal = temp_clone->LegalActions();
          } else {
            bool has_graft = false;
            for (Action ga : next_legal) {
              if (ga >= kActionSelectGraftPartner0 && ga < kActionSelectGraftPartner0 + 256) {
                has_graft = true;
                break;
              }
            }
            if (has_graft) {
              Action graft_act = base_evaluator_->SelectAction(*temp_clone, player, /*greedy=*/false, temperature, prop_policy_rng);
              temp_clone->ApplyAction(graft_act);
              rec.proposed_action_chain.push_back(graft_act);
              rec.proposed_action_names.push_back(DiagnosticActionToString(graft_act));
              AdvanceThroughChanceAndAcks(temp_clone.get(), prop_chance_rng);
              next_legal = temp_clone->LegalActions();
            }
          }
        }

        // Space selection
        if (temp_clone->CurrentPlayer() == player && !next_legal.empty()) {
          Action a_space = base_evaluator_->SelectAction(*temp_clone, player, /*greedy=*/false, temperature, prop_policy_rng);
          rec.proposed_action_chain.push_back(a_space);
          rec.proposed_action_names.push_back(DiagnosticActionToString(a_space));

          // Test whether this chain is legal in the REAL state (without Windfall)
          auto test_real = state.Clone();
          auto* d_real = dynamic_cast<DuneImperiumState*>(test_real.get());
          bool chain_legal_in_real = false;

          if (ContainsAction(d_real->LegalActions(), a1)) {
            d_real->ApplyAction(a1);
            AdvanceThroughChanceAndAcks(d_real, prop_chance_rng);
            auto real_legal_2 = d_real->LegalActions();

            for (size_t k = 1; k + 1 < rec.proposed_action_chain.size(); ++k) {
              Action step_act = rec.proposed_action_chain[k];
              if (ContainsAction(real_legal_2, step_act)) {
                d_real->ApplyAction(step_act);
                AdvanceThroughChanceAndAcks(d_real, prop_chance_rng);
                real_legal_2 = d_real->LegalActions();
              }
            }

            if (ContainsAction(real_legal_2, a_space)) {
              chain_legal_in_real = true;
            }
          }

          if (!chain_legal_in_real) {
            // Plan REQUIRES Windfall!
            rec.plan_requires_windfall = true;
            rec.windfall_reason = "Board space " + DiagnosticActionToString(a_space) + " requires solari from Windfall";
            rec.action_taken = kActionPlayWindfall;
            rec.action_taken_name = "PlayPlotIntrigue(Windfall)";
            rec.windfall_played = true;
            rec.solari_after = rec.actual_solari + 2;
            windfall_played_this_run_ = true;
            decision_records_.push_back(rec);
            return kActionPlayWindfall;
          } else {
            // Space is already affordable without Windfall! Retain Windfall!
            rec.plan_requires_windfall = false;
            rec.windfall_reason = "Board space " + DiagnosticActionToString(a_space) + " is already affordable without Windfall";
            rec.action_taken = a1;
            rec.action_taken_name = DiagnosticActionToString(a1);
            rec.windfall_played = false;
            rec.solari_after = rec.actual_solari;
            decision_records_.push_back(rec);
            RecordResourceFollowUp(a1);
            return a1;
          }
        }
        if (ContainsAction(legal, a1)) {
          rec.plan_requires_windfall = false;
          rec.windfall_reason = "Card selection without identified solari requirement; retaining Windfall";
          rec.action_taken = a1;
          rec.action_taken_name = DiagnosticActionToString(a1);
          rec.windfall_played = false;
          rec.solari_after = rec.actual_solari;
          decision_records_.push_back(rec);
          RecordResourceFollowUp(a1);
          return a1;
        }
      }

      // Case B: a1 is not an agent card selection (e.g. EndTurn, Reveal, or purchase/tech)
      if (a1 == kActionEndTurn) {
        rec.plan_requires_windfall = false;
        rec.windfall_reason = "Plan ends turn without requiring solari";
        rec.action_taken = kActionEndTurn;
        rec.action_taken_name = "EndTurn";
        rec.windfall_played = false;
        rec.solari_after = rec.actual_solari;
        decision_records_.push_back(rec);
        return kActionEndTurn;
      }

      if (a1 == kActionReveal) {
        rec.plan_requires_windfall = false;
        rec.windfall_reason = "Plan reveals without requiring solari";
        rec.action_taken = kActionReveal;
        rec.action_taken_name = "Reveal";
        rec.windfall_played = false;
        rec.solari_after = rec.actual_solari;
        decision_records_.push_back(rec);
        return kActionReveal;
      }

      // Purchase or tech action: test legality in real state
      if (ContainsAction(legal, a1)) {
        rec.plan_requires_windfall = false;
        rec.windfall_reason = "Action " + DiagnosticActionToString(a1) + " is already affordable without Windfall";
        rec.action_taken = a1;
        rec.action_taken_name = DiagnosticActionToString(a1);
        rec.windfall_played = false;
        rec.solari_after = rec.actual_solari;
        decision_records_.push_back(rec);
        RecordResourceFollowUp(a1);
        return a1;
      } else {
        rec.plan_requires_windfall = true;
        rec.windfall_reason = "Action " + DiagnosticActionToString(a1) + " requires solari from Windfall";
        rec.action_taken = kActionPlayWindfall;
        rec.action_taken_name = "PlayPlotIntrigue(Windfall)";
        rec.windfall_played = true;
        rec.solari_after = rec.actual_solari + 2;
        windfall_played_this_run_ = true;
        decision_records_.push_back(rec);
        return kActionPlayWindfall;
      }

    } catch (const std::exception& e) {
      rec.fallback_active = true;
      rec.fallback_reason = std::string("Exception during proposal simulation: ") + e.what();
      Action act = base_evaluator_->SelectAction(state, player, greedy, temperature, rng);
      rec.action_taken = act;
      rec.action_taken_name = DiagnosticActionToString(act);
      rec.windfall_played = (act == kActionPlayWindfall);
      rec.solari_after = rec.windfall_played ? (rec.actual_solari + 2) : rec.actual_solari;
      if (rec.windfall_played) windfall_played_this_run_ = true;
      decision_records_.push_back(rec);
      RecordResourceFollowUp(act);
      return act;
    }
  }

 private:
  IPolicyDiagnosticsEvaluator* base_evaluator_ = nullptr;
  int designated_holder_ = 0;
  int candidate_id_ = -1;
  int replicate_id_ = -1;
  int turn_step_ = 0;
  bool windfall_played_this_run_ = false;
  bool resource_followup_recorded_ = false;
  Action last_resource_action_ = open_spiel::kInvalidAction;
  std::string last_resource_action_name_;
  std::vector<WindfallDecisionRecord> decision_records_;
};

} // namespace dune_diagnostics
} // namespace open_spiel

#endif // OPEN_SPIEL_EXAMPLES_DUNE_FROZEN_POLICY_DIAG_EVALUATORS_H_
