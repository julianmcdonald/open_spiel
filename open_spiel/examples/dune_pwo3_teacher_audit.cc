// PWO-3 Phases 2+3: teacher-audit search runner and rollout oracle.
//
// Implements docs/PWO3_REGISTRATION.md sections 2, 4, 6.2, 7.2, 7.3.
//
// Modes:
//   --mode=search             Branch-A x tiers x seeds over the corpora   -> JSONL
//   --mode=select             the registered seeded subsets (section 3.2) -> JSON
//   --mode=oracle             oracle values for a REGISTERED demand list  -> JSONL
//   --mode=oracle_sanity      conversion-stratum two-block Spearman check -> JSON
//
// ---------------------------------------------------------------------------
// FORKED FROM dune_pwo2_qualify.cc, DELIBERATELY, RATHER THAN EXTENDING IT
// ---------------------------------------------------------------------------
// Extending the PWO-2 tool in place would change the binary that PWO-2's committed
// artifacts are attributed to and make them non-reproducible. This fork leaves
// dune_pwo2_qualify byte-identical. The shared SEARCH CORE is necessarily common
// to both -- that is exactly what the section 6.1.1 cross-build reproduction gate
// tests, on all 384 Branch-A fixed_800 rows.
//
// Two PWO-2 design decisions are inherited verbatim and are not cosmetic:
//
// 1. PER-ROOT FRESH FULL SEARCH, NOT A SESSION. DuneSearchSession routes kPurchase
//    and kCombatIntrigue roots through the SHORT-WINDOW budget
//    (dune_search_session.cc:344), so a session-driven runner would silently give
//    every purchase/combat root 16 simulations at EVERY tier. Driving
//    DunePUCTISMCTSBot::RunSearch directly makes each tier mean what it says at
//    every root, and it is what PWO-2 section 12.6 requirement 1 demands: isolated
//    roots, so a label has a single clean provenance.
//
// 2. SELECTION IS ARGMAX, INCLUDING ON THE FALLBACK PATH. DunePUCTISMCTSBot::Step()
//    SAMPLES from result.policy. Under the registered --temperature=0.0 the searched
//    policy is one-hot, so sampling is a no-op there -- but on a STARVED root
//    result.policy is the raw network prior, which is not one-hot, and Step() would
//    sample it. Sampling a raw prior under a temperature-0 pin is the documented
//    below-parity class (2026-07-12 / 2026-07-20). This runner selects by
//    deterministic argmax with a lowest-action-id tie-break.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/json.h"
#include <torch/torch.h>

#include "dune_evaluator.h"
#include "dune_network.h"
#include "dune_puct_is_mcts.h"
#include "dune_pwo2_common.h"
#include "dune_pwo3_common.h"
#include "dune_search_routing.h"
#include "dune_search_session.h"  // SampleActionFromPrior(prior, r_val)
#include "dune_seed_utils.h"
#include "dune_sha256.h"

ABSL_FLAG(std::string, mode, "search",
          "search | select | oracle | oracle_sanity");
ABSL_FLAG(std::string, main_corpus, "data/pwo2_root_corpus.json", "");
ABSL_FLAG(std::string, conversion_corpus, "", "PWO-3 conversion stratum (may be empty).");
ABSL_FLAG(std::string, sm_corpus, "", "PWO-3 SM stratum (may be empty). SEARCH ONLY.");
ABSL_FLAG(std::string, output_path, "", "Output JSONL/JSON path.");
ABSL_FLAG(std::string, branch_a_checkpoint,
          "artifacts/branch_a_frozen/branch_a_seed11_model_update_2450.pt",
          "The teacher. The ONLY model ever searched in this WO.");
ABSL_FLAG(int, hidden_dim, 2048, "");
ABSL_FLAG(int, num_blocks, 8, "");
ABSL_FLAG(bool, nonlinear_value_head, false, "");
ABSL_FLAG(int, threads, 1, "Worker threads.");

// ---- The inherited controller pin (registration section 2). Every flag is passed
// explicitly by launch.sh; no default here defines a measurement. ---------------
ABSL_FLAG(int, opponent_mode, 1, "0=kMaxN, 1=policy sampling. Pin: 1.");
ABSL_FLAG(double, simulated_opponent_temperature, 1.0, "Pin: 1.0");
ABSL_FLAG(double, temperature, 0.0, "Pin: 0.0 (greedy)");
ABSL_FLAG(double, puct_c, 0.3, "Pin: 0.3");
ABSL_FLAG(double, root_prior_temperature, 1.0, "Pin: 1.0");
ABSL_FLAG(double, utility_divisor, 4.0, "Pin: 4.0");
ABSL_FLAG(int, max_world_samples, -1, "Pin: -1");
ABSL_FLAG(double, dirichlet_epsilon, 0.0, "Pin: 0.0 (root noise off)");
ABSL_FLAG(bool, check_strategic_state, false, "Pin: false");
ABSL_FLAG(double, candidate_logit_cap, 10.0, "Pin: 10.0");
ABSL_FLAG(int, max_nodes, 200000, "Pin: 200000 for fixed tiers.");
ABSL_FLAG(int, min_visit_threshold, 2, "Pin: 2.");
// Registration section 2.2: PWO-2 left these at their struct defaults. PWO-3 makes
// them explicit at exactly those values -- behaviour-neutral by construction, and
// the section 6.1.1 gate proves it.
ABSL_FLAG(double, covered_prior_threshold, 0.50, "Pin: 0.50 (was an implicit default).");
ABSL_FLAG(double, stability_checkpoint_fraction, 0.5,
          "Pin: 0.5 (was an implicit default; the half-budget checkpoint).");

// ---- Grid selection -------------------------------------------------------
ABSL_FLAG(std::string, tiers, "fixed_800", "fixed_400 | fixed_800 | live_50");
ABSL_FLAG(std::string, search_seeds, "1301,1302", "");
ABSL_FLAG(std::string, corpora, "main,conversion,sm", "Which corpora to search.");
ABSL_FLAG(std::string, root_filter_path, "",
          "JSON array of root_ids to restrict to (registered subsets, gates, pilots).");
ABSL_FLAG(int, root_limit, 0, "Cap roots in canonical order for pilots. 0 = all.");
ABSL_FLAG(int, live_max_simulations, 100000,
          "Post-Amendment-2 ratified live ceiling. The escalation is SPENT.");
ABSL_FLAG(int, live_max_nodes, 800000, "Post-Amendment-2 ratified live node ceiling.");
ABSL_FLAG(double, live_deadline_ms, 50000.0, "Live tier wall-clock deadline.");

// ---- Oracle ---------------------------------------------------------------
ABSL_FLAG(std::string, demand_list_path, "",
          "JSON array of {root_id, action} -- the REGISTERED demand list (section 7.2).");
ABSL_FLAG(int, oracle_continuations, 128, "Registered: 128.");
ABSL_FLAG(int, oracle_continuation_offset, 0, "Block offset (sanity uses 0 and 128).");
ABSL_FLAG(int, sanity_roots, 8, "Registration 7.3: 8 conversion roots.");
ABSL_FLAG(int, sanity_max_actions, 5, "Registration 7.3: min(n_legal, 5).");

using namespace open_spiel;

namespace {

struct Root {
  std::string root_id;
  std::string stratum;   // decision-role stratum; PWO-2 meaning on EVERY root
  std::string corpus;    // main | conversion | sm
  std::string source_arm;
  std::string half;
  int source_episode_id = 0;
  int round = 0;
  int decision_index = 0;
  Player player = kInvalidPlayer;
  std::vector<Action> history;
  std::vector<Action> legal_actions;
};

struct Tier {
  std::string name;
  int max_sims;
  double time_budget_ms;
  int max_nodes;
  bool live;
};

Tier ResolveTier(const std::string& name) {
  const double kInf = std::numeric_limits<double>::infinity();
  const int fixed_nodes = absl::GetFlag(FLAGS_max_nodes);
  if (name == "fixed_400") return {"fixed_400", 400, kInf, fixed_nodes, false};
  if (name == "fixed_800") return {"fixed_800", 800, kInf, fixed_nodes, false};
  if (name == "live_50")
    return {"live_50", absl::GetFlag(FLAGS_live_max_simulations),
            absl::GetFlag(FLAGS_live_deadline_ms),
            absl::GetFlag(FLAGS_live_max_nodes), true};
  SpielFatalError("unknown tier: " + name);
}

std::vector<std::string> SplitCsv(const std::string& s) {
  std::vector<std::string> out;
  if (s.empty()) return out;
  for (absl::string_view p : absl::StrSplit(s, ',')) {
    if (!p.empty()) out.emplace_back(p);
  }
  return out;
}

std::unique_ptr<State> Reconstruct(const std::shared_ptr<const Game>& game,
                                   const std::vector<Action>& history) {
  auto st = game->NewInitialState();
  for (Action a : history) st->ApplyAction(a);
  return st;
}

std::shared_ptr<SharedDunePolicyValueNetImpl> LoadModel(
    const std::string& path, int64_t obs, int64_t act, torch::Device dev) {
  auto m = std::make_shared<SharedDunePolicyValueNetImpl>(
      obs, absl::GetFlag(FLAGS_hidden_dim), act, absl::GetFlag(FLAGS_num_blocks),
      absl::GetFlag(FLAGS_nonlinear_value_head));
  try {
    torch::load(m, path, dev);
  } catch (const std::exception& e) {
    std::cerr << "Failed to load " << path << ": " << e.what() << "\n";
    std::exit(1);
  }
  m->to(dev);
  m->eval();
  return m;
}

DuneSearchConfig MakePinnedConfig(const Tier& tier, uint64_t search_rng_seed) {
  DuneSearchConfig c;
  c.max_simulations = tier.max_sims;
  c.relative_time_budget_ms = tier.time_budget_ms;
  c.max_nodes = tier.max_nodes;
  c.puct_c = absl::GetFlag(FLAGS_puct_c);
  const int om = absl::GetFlag(FLAGS_opponent_mode);
  SPIEL_CHECK_TRUE(om == 0 || om == 1);
  c.opponent_mode = om == 0 ? SearchOpponentMode::kMaxN : SearchOpponentMode::kPolicy;
  c.temperature = absl::GetFlag(FLAGS_temperature);
  c.opponent_temperature = absl::GetFlag(FLAGS_simulated_opponent_temperature);
  c.max_world_samples = absl::GetFlag(FLAGS_max_world_samples);
  c.utility_divisor = absl::GetFlag(FLAGS_utility_divisor);
  c.min_visit_threshold = absl::GetFlag(FLAGS_min_visit_threshold);
  c.dirichlet_epsilon = absl::GetFlag(FLAGS_dirichlet_epsilon);
  c.check_strategic_state = absl::GetFlag(FLAGS_check_strategic_state);
  c.root_prior_temperature = absl::GetFlag(FLAGS_root_prior_temperature);
  // Registration 2.2: explicit at the values PWO-2 took implicitly.
  c.covered_prior_threshold = absl::GetFlag(FLAGS_covered_prior_threshold);
  c.conservative_stability_checkpoint_fraction =
      absl::GetFlag(FLAGS_stability_checkpoint_fraction);
  c.seed = search_rng_seed;
  // Session-only knobs. Unconsulted here (design note 1); set to their registered
  // no-op values so a manifest diff never suggests otherwise. The live tier's
  // reserve is 0.0 per the PWO-2 section 2.3 registered divergence -- the runner
  // does a standalone root search with no continuations, so reserving 10 s for
  // continuations that will never be issued would make "50 s" false.
  c.fixed_continuation_reserve = 0;
  c.purchase_combat_budget = 16;
  c.live_continuation_reserve_seconds = 0.0;
  c.fixed_session_limit = tier.max_sims;
  return c;
}

// Registration 14.2: lowest-action-id tie-break, everywhere.
Action ArgmaxPolicy(const ActionsAndProbs& policy) {
  Action best = kInvalidAction;
  double best_p = -1.0;
  for (const auto& ap : policy) {
    if (ap.second > best_p || (ap.second == best_p && ap.first < best)) {
      best_p = ap.second;
      best = ap.first;
    }
  }
  return best;
}

double RawPolicyRollout(const State& start, Player owner,
                        algorithms::Evaluator* evaluator, uint64_t seed,
                        double utility_divisor) {
  std::mt19937 rng = dune_seed::MakeRng32(seed);
  auto state = start.Clone();
  while (!state->IsTerminal()) {
    if (state->IsChanceNode()) {
      auto outcomes = state->ChanceOutcomes();
      state->ApplyAction(SampleAction(outcomes, absl::Uniform(rng, 0.0, 1.0)).first);
    } else if (state->CurrentPlayer() == kSimultaneousPlayerId) {
      std::vector<Action> joint;
      for (int p = 0; p < state->NumPlayers(); ++p) {
        auto acts = state->LegalActions(p);
        std::uniform_int_distribution<int> d(0, acts.size() - 1);
        joint.push_back(acts[d(rng)]);
      }
      state->ApplyActions(joint);
    } else {
      ActionsAndProbs prior = evaluator->Prior(*state);
      state->ApplyAction(SampleActionFromPrior(prior, absl::Uniform(rng, 0.0, 1.0)));
    }
  }
  return state->Returns()[owner] / utility_divisor;
}

void LoadCorpusInto(const std::string& path, const std::string& corpus_tag,
                    std::vector<Root>* out) {
  if (path.empty()) return;
  std::ifstream f(path);
  if (!f) SpielFatalError("cannot open corpus: " + path);
  std::string content((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
  auto parsed = json::FromString(content);
  if (!parsed) SpielFatalError("corpus is not valid JSON: " + path);
  for (const auto& v : parsed.value().GetArray()) {
    auto o = v.GetObject();
    Root r;
    r.root_id = o.at("root_id").GetString();
    r.stratum = o.at("stratum").GetString();
    // The main corpus predates the `corpus` field; tag it here.
    r.corpus = (o.find("corpus") != o.end()) ? o.at("corpus").GetString() : corpus_tag;
    SPIEL_CHECK_EQ(r.corpus, corpus_tag);
    r.source_arm = o.at("source_arm").GetString();
    r.half = o.at("half").GetString();
    r.source_episode_id = static_cast<int>(o.at("source_episode_id").GetInt());
    r.round = static_cast<int>(o.at("round").GetInt());
    r.decision_index = static_cast<int>(o.at("decision_index").GetInt());
    r.player = static_cast<Player>(o.at("player").GetInt());
    for (const auto& a : o.at("history").GetArray())
      r.history.push_back(static_cast<Action>(a.GetInt()));
    for (const auto& a : o.at("legal_actions").GetArray())
      r.legal_actions.push_back(static_cast<Action>(a.GetInt()));
    // Fail closed: the corpus must still replay under the frozen engine.
    SPIEL_CHECK_EQ(pwo2::HistoryHash(r.history), r.root_id);
    SPIEL_CHECK_TRUE(pwo3::IsAscending(r.legal_actions));
    out->push_back(std::move(r));
  }
}

std::string FmtExact(double v) { return absl::StrFormat("%.17g", v); }

json::Array ToJsonArray(const std::vector<Action>& v) {
  json::Array a;
  for (Action x : v) a.push_back(static_cast<int64_t>(x));
  return a;
}

json::Array ToJsonArray(const std::vector<int>& v) {
  json::Array a;
  for (int x : v) a.push_back(static_cast<int64_t>(x));
  return a;
}

json::Array ToJsonArray(const std::vector<double>& v) {
  json::Array a;
  for (double x : v) a.push_back(x);
  return a;
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  at::set_num_threads(1);

  const std::string mode = absl::GetFlag(FLAGS_mode);
  const int num_threads = absl::GetFlag(FLAGS_threads);
  const double util_div = absl::GetFlag(FLAGS_utility_divisor);
  const std::string self_sha = pwo2::Sha256File(argv[0]);

  auto game = LoadGame("dune_imperium");
  const int64_t obs_size = game->GetType().provides_information_state_tensor
                               ? game->InformationStateTensorSize()
                               : game->ObservationTensorSize();
  const int64_t act_size = game->NumDistinctActions();
  torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                                     : torch::Device(torch::kCPU);

  const std::string ck_a = absl::GetFlag(FLAGS_branch_a_checkpoint);
  auto model_a = LoadModel(ck_a, obs_size, act_size, device);
  const std::string sha_a = pwo2::Sha256File(ck_a);
  const float cap = static_cast<float>(absl::GetFlag(FLAGS_candidate_logit_cap));

  std::cerr << "dune_pwo3_teacher_audit mode=" << mode << " threads=" << num_threads
            << " device=" << (device.is_cuda() ? "CUDA" : "CPU")
            << "\n  binary sha256 " << self_sha
            << "\n  branch_a      " << sha_a
            << "\n  Fnv1a64(PWO2_SEARCH)=" << pwo2::kDomainSearch
            << " Fnv1a64(PWO2_ORACLE)=" << pwo2::kDomainOracle
            << " new_selection_seed=" << pwo3::kNewSelectionSeed << "\n";

  std::vector<Root> roots;
  const auto want_corpora = SplitCsv(absl::GetFlag(FLAGS_corpora));
  auto wants = [&](const std::string& c) {
    return std::find(want_corpora.begin(), want_corpora.end(), c) != want_corpora.end();
  };
  if (wants("main")) LoadCorpusInto(absl::GetFlag(FLAGS_main_corpus), "main", &roots);
  if (wants("conversion"))
    LoadCorpusInto(absl::GetFlag(FLAGS_conversion_corpus), "conversion", &roots);
  if (wants("sm")) LoadCorpusInto(absl::GetFlag(FLAGS_sm_corpus), "sm", &roots);

  // Registered subsets (the live-audit sample, the gates, the pilots) are named by
  // an explicit root-id list, never by an ad-hoc predicate here.
  if (!absl::GetFlag(FLAGS_root_filter_path).empty()) {
    std::ifstream ff(absl::GetFlag(FLAGS_root_filter_path));
    if (!ff) SpielFatalError("cannot open --root_filter_path");
    std::string c((std::istreambuf_iterator<char>(ff)), std::istreambuf_iterator<char>());
    auto p = json::FromString(c);
    if (!p) SpielFatalError("--root_filter_path is not valid JSON");
    std::set<std::string> keep;
    for (const auto& v : p.value().GetArray()) keep.insert(v.GetString());
    std::vector<Root> f;
    for (auto& r : roots)
      if (keep.count(r.root_id)) f.push_back(r);
    if (f.size() != keep.size()) {
      std::cerr << "STOP: --root_filter_path named " << keep.size()
                << " root ids but only " << f.size()
                << " were found in the loaded corpora.\n";
      return 1;
    }
    roots.swap(f);
  }
  if (absl::GetFlag(FLAGS_root_limit) > 0 &&
      roots.size() > static_cast<size_t>(absl::GetFlag(FLAGS_root_limit))) {
    roots.resize(absl::GetFlag(FLAGS_root_limit));
  }
  std::cerr << "  roots: " << roots.size() << "\n";

  std::ofstream out(absl::GetFlag(FLAGS_output_path));
  if (!out) {
    std::cerr << "cannot open --output_path\n";
    return 1;
  }
  std::mutex out_mu;

  // =========================================================================
  // MODE: search
  // =========================================================================
  if (mode == "search") {
    struct Job {
      int root;
      Tier tier;
      int seed;
    };
    std::vector<Job> jobs;
    for (const auto& tname : SplitCsv(absl::GetFlag(FLAGS_tiers))) {
      Tier t = ResolveTier(tname);
      for (size_t i = 0; i < roots.size(); ++i) {
        // The SM stratum is seed-1301 only and never live (registration 5.3).
        for (const auto& s : SplitCsv(absl::GetFlag(FLAGS_search_seeds))) {
          const int seed = std::stoi(s);
          if (roots[i].corpus == "sm" && (seed != 1301 || t.live)) continue;
          jobs.push_back({static_cast<int>(i), t, seed});
        }
      }
    }
    std::cerr << "  jobs: " << jobs.size() << "\n";

    std::atomic<int> next{0};
    std::atomic<int> done{0};
    auto worker = [&]() {
      auto ev_a = std::make_shared<DuneNNEvaluator>(model_a, device, cap);
      while (true) {
        int ji = next.fetch_add(1);
        if (ji >= static_cast<int>(jobs.size())) break;
        const Job& j = jobs[ji];
        const Root& r = roots[j.root];
        auto state = Reconstruct(game, r.history);
        SPIEL_CHECK_EQ(state->CurrentPlayer(), r.player);

        std::shared_ptr<algorithms::Evaluator> ev =
            std::static_pointer_cast<algorithms::Evaluator>(ev_a);

        const uint64_t rng_seed = pwo2::SearchRngSeed(j.seed, r.root_id);
        DuneSearchConfig cfg = MakePinnedConfig(j.tier, rng_seed);
        DunePUCTISMCTSBot bot(cfg, ev);
        DuneSearchResult res =
            bot.RunSearch(*state, j.tier.max_sims, j.tier.time_budget_ms, 0);
        const Action deployed = ArgmaxPolicy(res.policy);
        SearchDiagnostics d =
            bot.GetRootDiagnostics(*state, cfg.min_visit_threshold, deployed);

        // ---- Registration 4.1: three actions, as SEPARATE fields -----------
        const int n_legal = static_cast<int>(d.actions.size());
        SPIEL_CHECK_EQ(n_legal, static_cast<int>(r.legal_actions.size()));
        // ASSERT 1: ascending root_actions is what makes the lowest-action-id
        // tie-break identical to the search core's "first in legal order".
        SPIEL_CHECK_TRUE(pwo3::IsAscending(d.actions));
        const bool raw_from_raw_priors =
            d.raw_priors.size() == d.actions.size();
        const std::vector<double>& raw_prior =
            raw_from_raw_priors ? d.raw_priors : d.priors;
        SPIEL_CHECK_EQ(raw_prior.size(), d.actions.size());
        SPIEL_CHECK_EQ(d.visit_counts.size(), d.actions.size());

        const Action raw_argmax = pwo3::ArgmaxLowestId(d.actions, raw_prior);
        const Action pre_gate = pwo3::ArgmaxLowestId(d.actions, d.visit_counts);

        // ---- The deployed policy VECTOR, reindexed onto root_actions -------
        // Registration 11.2: this vector, not any flag, is P-post's q_search.
        // ASSERT 2: the action sets must match.
        std::map<Action, double> pol;
        for (const auto& ap : res.policy) pol[ap.first] = ap.second;
        SPIEL_CHECK_EQ(pol.size(), d.actions.size());
        std::vector<double> deployed_policy;
        deployed_policy.reserve(n_legal);
        for (Action a : d.actions) {
          auto it = pol.find(a);
          SPIEL_CHECK_TRUE(it != pol.end());
          deployed_policy.push_back(it->second);
        }
        // ASSERT 3: argmax(deployed_policy) == deployed_action.
        SPIEL_CHECK_EQ(pwo3::ArgmaxLowestId(d.actions, deployed_policy), deployed);

        // ---- The coverage gate, from its own INPUTS, never from a flag ------
        const bool gate_fired = pwo3::CoverageGateFires(
            d.num_covered_actions, d.covered_prior_mass, n_legal);
        // The two controller invariants (registration 14.4), verified with 0
        // violations on all 2,688 PWO-2 rows. A violation here means the search
        // core changed under the rebuild and must not be measured over.
        if (gate_fired) {
          SPIEL_CHECK_EQ(deployed, raw_argmax);
        } else {
          SPIEL_CHECK_EQ(deployed, pre_gate);
        }
        // ASSERT 5: isolated roots inherit nothing.
        SPIEL_CHECK_EQ(res.diagnostics.inherited_root_visits, 0);

        json::Object o;
        o["root_id"] = r.root_id;
        o["stratum"] = r.stratum;
        o["corpus"] = r.corpus;
        o["source_arm"] = r.source_arm;
        o["source_episode_id"] = static_cast<int64_t>(r.source_episode_id);
        o["half"] = r.half;
        o["round"] = static_cast<int64_t>(r.round);
        o["decision_index"] = static_cast<int64_t>(r.decision_index);
        o["acting_player"] = static_cast<int64_t>(r.player);
        o["n_legal"] = static_cast<int64_t>(n_legal);
        o["tier"] = j.tier.name;
        o["seed"] = static_cast<int64_t>(j.seed);
        o["search_rng_seed"] = static_cast<int64_t>(rng_seed);

        // PWO-2 field name retained so the section 6.1.1 gate compares like with like.
        o["chosen_action"] = static_cast<int64_t>(deployed);
        o["deployed_action"] = static_cast<int64_t>(deployed);
        o["raw_argmax_action"] = static_cast<int64_t>(raw_argmax);
        o["pre_gate_visit_argmax_action"] = static_cast<int64_t>(pre_gate);
        o["raw_prior_vector"] = ToJsonArray(raw_prior);
        o["raw_prior_source"] = std::string(raw_from_raw_priors ? "raw_priors" : "priors");
        o["deployed_policy"] = ToJsonArray(deployed_policy);
        o["visit_counts"] = ToJsonArray(d.visit_counts);
        o["root_actions"] = ToJsonArray(d.actions);
        o["coverage_gate_fired"] = gate_fired;
        o["coverage_gate_changed_action"] = gate_fired && (deployed != pre_gate);
        o["zero_root_visits"] = d.total_root_visits == 0;

        // ---- PWO-2 telemetry, kept unchanged -------------------------------
        o["simulations_completed"] = static_cast<int64_t>(res.simulations_completed);
        o["elapsed_time_ms"] = res.elapsed_time_ms;
        o["used_fallback"] = res.used_fallback;
        o["timeout_status"] = res.timeout_status;
        o["fallback_reason"] = res.fallback_reason;
        o["inference_count"] = static_cast<int64_t>(res.inference_count);
        o["total_root_visits"] = static_cast<int64_t>(d.total_root_visits);
        o["num_covered_actions"] = static_cast<int64_t>(d.num_covered_actions);
        o["covered_prior_mass"] = d.covered_prior_mass;
        o["tree_node_count"] = static_cast<int64_t>(d.unique_nodes);
        o["max_depth"] = d.max_depth;
        o["mean_depth"] = d.mean_depth;
        o["terminal_leaf_fraction"] = d.terminal_leaf_fraction;
        o["chosen_action_raw_prior_probability"] = d.chosen_action_raw_prior_probability;
        o["chosen_action_raw_prior_rank"] =
            static_cast<int64_t>(d.chosen_action_raw_prior_rank);
        o["action_changed_vs_raw_argmax"] = d.action_changed_vs_raw_argmax;

        // ---- Registration 4.2: the two half-budget checkpoints --------------
        o["half_budget_checkpoint_sim"] =
            static_cast<int64_t>(std::floor(
                j.tier.max_sims * absl::GetFlag(FLAGS_stability_checkpoint_fraction)));
        o["half_budget_checkpoint_reached"] =
            res.diagnostics.stability_checkpoint_reached;
        o["half_budget_visit_argmax_action"] =
            static_cast<int64_t>(res.diagnostics.stability_checkpoint_action);
        o["half_budget_num_covered_actions"] =
            static_cast<int64_t>(res.diagnostics.stability_checkpoint_num_covered_actions);
        o["half_budget_covered_prior_mass"] =
            res.diagnostics.stability_checkpoint_covered_prior_mass;
        o["half_time_checkpoint_reached"] = res.diagnostics.half_time_checkpoint_reached;
        o["half_time_checkpoint_sim"] =
            static_cast<int64_t>(res.diagnostics.half_time_checkpoint_sim);
        o["half_time_visit_argmax_action"] =
            static_cast<int64_t>(res.diagnostics.half_time_checkpoint_action);
        o["half_time_num_covered_actions"] =
            static_cast<int64_t>(res.diagnostics.half_time_checkpoint_num_covered_actions);
        o["half_time_covered_prior_mass"] =
            res.diagnostics.half_time_checkpoint_covered_prior_mass;

        // ---- Registration 4.3: the five-quantity accounting -----------------
        o["new_simulations"] = static_cast<int64_t>(res.simulations_completed);
        o["inherited_root_visits"] =
            static_cast<int64_t>(res.diagnostics.inherited_root_visits);
        o["budget_fallback"] =
            (res.simulations_completed == 0 && res.fallback_reason == "timeout");
        o["budget_terminated"] =
            (res.fallback_reason == "timeout" || res.fallback_reason == "max_nodes");
        o["coverage_fallback"] = gate_fired;
        o["zero_visits_fallback"] = (res.fallback_reason == "zero_visits");
        o["executed_action"] = static_cast<int64_t>(deployed);

        // ---- Registration 4.4: conversion / SM tagging (real range 741-752) --
        const auto legal_conv = pwo3::LegalConversionActions(d.actions);
        o["conversion_legal"] = !legal_conv.empty();
        o["legal_conversion_actions"] = ToJsonArray(legal_conv);
        o["n_legal_conversion_actions"] = static_cast<int64_t>(legal_conv.size());
        auto amt = [](Action a) -> json::Value {
          const int v = pwo3::ConversionAmount(a);
          return v < 0 ? json::Value() : json::Value(static_cast<int64_t>(v));
        };
        o["chosen_conversion_amount"] = amt(deployed);
        o["raw_argmax_conversion_amount"] = amt(raw_argmax);
        o["pre_gate_conversion_amount"] = amt(pre_gate);
        o["sm_legal"] = pwo3::HasSwordmaster(d.actions);
        o["chose_sm"] = (deployed == pwo3::kSwordmasterAction);
        o["raw_argmax_is_sm"] = (raw_argmax == pwo3::kSwordmasterAction);
        o["pre_gate_is_sm"] = (pre_gate == pwo3::kSwordmasterAction);

        o["configured_max_simulations"] = static_cast<int64_t>(j.tier.max_sims);
        o["configured_max_nodes"] = static_cast<int64_t>(cfg.max_nodes);
        o["configured_time_budget_ms"] =
            std::isinf(j.tier.time_budget_ms) ? -1.0 : j.tier.time_budget_ms;
        o["configured_covered_prior_threshold"] = cfg.covered_prior_threshold;
        o["binary_sha256"] = self_sha;
        o["model_sha256"] = sha_a;

        std::lock_guard<std::mutex> lk(out_mu);
        out << json::ToString(o) << "\n";
        int n = ++done;
        if (n % 25 == 0 || n == static_cast<int>(jobs.size()))
          std::cerr << "  " << n << "/" << jobs.size() << "\r" << std::flush;
      }
    };
    std::vector<std::thread> ws;
    for (int i = 0; i < num_threads; ++i) ws.emplace_back(worker);
    for (auto& w : ws) w.join();
    std::cerr << "\n  search done: " << done.load() << " rows\n";
    return 0;
  }

  // =========================================================================
  // MODE: select -- emit the registered seeded subsets (registration 3.2).
  //
  // Every seeded selection in this WO happens HERE, in C++, and is emitted as a
  // sorted JSON array of root ids. DeriveSeed is trivially portable, but
  // MakeRng64 (std::mt19937_64) and std::shuffle are libstdc++-specific, so a
  // Python reimplementation would be a second, silently divergent instrument for
  // a registered selection. One source of truth, emitted as an artifact.
  // =========================================================================
  if (mode == "select") {
    // Canonical order for every candidate list: (source_arm, source_episode_id,
    // decision_index), matching the corpus's own final ordering.
    auto canonical = [&](std::vector<int> idx) {
      std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        if (roots[a].source_arm != roots[b].source_arm)
          return roots[a].source_arm < roots[b].source_arm;
        if (roots[a].source_episode_id != roots[b].source_episode_id)
          return roots[a].source_episode_id < roots[b].source_episode_id;
        return roots[a].decision_index < roots[b].decision_index;
      });
      return idx;
    };
    auto take = [&](std::vector<int> pool, uint64_t seed, int k) {
      pool = canonical(std::move(pool));
      auto rng = dune_seed::MakeRng64(seed);
      std::shuffle(pool.begin(), pool.end(), rng);
      if (static_cast<int>(pool.size()) > k) pool.resize(k);
      std::vector<std::string> ids;
      for (int i : pool) ids.push_back(roots[i].root_id);
      std::sort(ids.begin(), ids.end());
      return ids;
    };
    auto emit = [](const std::vector<std::string>& ids) {
      json::Array a;
      for (const auto& s : ids) a.push_back(s);
      return a;
    };

    json::Object sel;

    // (a) Live-audit AP subsample: 24 of the 32 calibration-half main-corpus
    //     agent_primary roots (tag 501).
    {
      std::vector<int> pool;
      for (size_t i = 0; i < roots.size(); ++i)
        if (roots[i].corpus == "main" && roots[i].stratum == "agent_primary" &&
            roots[i].half == "calibration")
          pool.push_back(i);
      std::cerr << "  live-audit AP pool (calibration half): " << pool.size() << "\n";
      sel["live_audit_agent_primary_24"] =
          emit(take(pool, pwo3::SubsampleSeed(pwo3::kTagLiveAuditSubsample), 24));
      // Registration 6.3 stage 2b: the REMAINING 8. The main corpus holds exactly
      // 32 per half and stage 1 consumes 24 -- there is no 16-root reserve.
      std::set<std::string> chosen;
      for (const auto& s : take(pool, pwo3::SubsampleSeed(pwo3::kTagLiveAuditSubsample), 24))
        chosen.insert(s);
      std::vector<std::string> rest;
      for (int i : canonical(pool))
        if (!chosen.count(roots[i].root_id)) rest.push_back(roots[i].root_id);
      std::sort(rest.begin(), rest.end());
      sel["live_audit_stage2b_remaining_8"] = emit(rest);
    }

    // (b) Oracle replication subset: 32 main-corpus roots, 8 per main stratum
    //     (tag 502, per-stratum stream = DeriveSeed(DeriveSeed(seed, 502), i)).
    {
      static const char* kStrata[4] = {"agent_primary", "agent_continuation",
                                       "purchase", "combat_intrigue"};
      std::vector<std::string> all;
      for (int si = 0; si < 4; ++si) {
        std::vector<int> pool;
        for (size_t i = 0; i < roots.size(); ++i)
          if (roots[i].corpus == "main" && roots[i].stratum == kStrata[si])
            pool.push_back(i);
        const uint64_t s = dune_seed::DeriveSeed(
            pwo3::SubsampleSeed(pwo3::kTagReplicationSubset),
            static_cast<uint64_t>(si));
        for (const auto& id : take(pool, s, 8)) all.push_back(id);
      }
      std::sort(all.begin(), all.end());
      sel["oracle_replication_subset_32"] = emit(all);
    }

    // (c) Conditional section 12.7 validation-half live sample: 16 of the 32
    //     validation-half main-corpus agent_primary roots (tag 504). FIXED HERE,
    //     at CP0a, like everything else -- it may only RUN after the freeze commit.
    {
      std::vector<int> pool;
      for (size_t i = 0; i < roots.size(); ++i)
        if (roots[i].corpus == "main" && roots[i].stratum == "agent_primary" &&
            roots[i].half == "validation")
          pool.push_back(i);
      sel["validation_half_live_sample_16"] =
          emit(take(pool, pwo3::SubsampleSeed(pwo3::kTagValidationLiveSample), 16));
    }

    // (d) The live tier's conversion-stratum roots: calibration half, role
    //     agent_primary or agent_continuation. DETERMINISTIC, not seeded --
    //     production gives purchase/combat decisions the 16-sim short-window
    //     budget, so a 50 s search there measures nothing production-relevant.
    {
      std::vector<std::string> ids;
      for (const auto& r : roots)
        if (r.corpus == "conversion" && r.half == "calibration" &&
            (r.stratum == "agent_primary" || r.stratum == "agent_continuation"))
          ids.push_back(r.root_id);
      std::sort(ids.begin(), ids.end());
      sel["live_audit_conversion_roots"] = emit(ids);
    }

    sel["new_selection_seed"] = static_cast<int64_t>(pwo3::kNewSelectionSeed);
    sel["binary_sha256"] = self_sha;
    sel["registration"] = std::string("docs/PWO3_REGISTRATION.md section 3.2");
    out << json::ToString(sel, true) << "\n";
    for (const char* k : {"live_audit_agent_primary_24",
                          "live_audit_stage2b_remaining_8",
                          "oracle_replication_subset_32",
                          "validation_half_live_sample_16",
                          "live_audit_conversion_roots"}) {
      std::cerr << "  " << k << ": " << sel[k].GetArray().size() << "\n";
    }
    return 0;
  }

  // =========================================================================
  // Oracle. The continuation policy is ALWAYS Branch-A u2450 raw at temperature
  // 1.0, for every root and every action (PWO-2 registration 4.4, inherited).
  // =========================================================================
  std::map<std::string, int> root_index;
  for (size_t i = 0; i < roots.size(); ++i) root_index[roots[i].root_id] = i;

  auto run_oracle_block = [&](const std::vector<std::pair<int, Action>>& reqs,
                              int offset, int n_cont, bool emit_rows,
                              std::vector<double>* values_out) {
    if (values_out) values_out->assign(reqs.size(), 0.0);
    std::atomic<size_t> next{0};
    std::atomic<int> done{0};
    auto worker = [&]() {
      auto ev = std::make_shared<DuneNNEvaluator>(model_a, device, cap);
      while (true) {
        size_t i = next.fetch_add(1);
        if (i >= reqs.size()) break;
        const Root& r = roots[reqs[i].first];
        const Action a = reqs[i].second;
        auto state = Reconstruct(game, r.history);
        state->ApplyAction(a);

        double sum = 0.0;
        std::vector<double> per;
        per.reserve(n_cont);
        for (int k = 0; k < n_cont; ++k) {
          const uint64_t s = pwo2::OracleContinuationSeed(r.root_id, offset + k);
          const double v = RawPolicyRollout(*state, r.player, ev.get(), s, util_div);
          sum += v;
          per.push_back(v);
        }
        const double value = sum / n_cont;
        if (values_out) (*values_out)[i] = value;

        if (emit_rows) {
          json::Object o;
          o["root_id"] = r.root_id;
          o["stratum"] = r.stratum;
          o["corpus"] = r.corpus;
          o["action"] = static_cast<int64_t>(a);
          o["value"] = value;
          o["n_continuations"] = static_cast<int64_t>(n_cont);
          o["continuation_offset"] = static_cast<int64_t>(offset);
          o["continuation_policy"] = std::string("branch_a_u2450_raw_temp1.0");
          o["binary_sha256"] = self_sha;
          o["model_sha256"] = sha_a;
          // Registration 4.5: EVERY new cell carries a digest. The PWO-2 cache's
          // missing digests are why the section 6.1.3 cross-build gate had to be
          // rebuilt on 12 preserved cells; do not repeat that.
          SHA256 h;
          for (double v : per) h.Update(FmtExact(v) + ";");
          o["per_continuation_digest"] = h.Final();
          std::lock_guard<std::mutex> lk(out_mu);
          out << json::ToString(o) << "\n";
        }
        int n = ++done;
        if (n % 10 == 0 || n == static_cast<int>(reqs.size()))
          std::cerr << "  oracle " << n << "/" << reqs.size() << "\r" << std::flush;
      }
    };
    std::vector<std::thread> ws;
    for (int i = 0; i < num_threads; ++i) ws.emplace_back(worker);
    for (auto& w : ws) w.join();
    std::cerr << "\n";
  };

  if (mode == "oracle") {
    const std::string dl = absl::GetFlag(FLAGS_demand_list_path);
    if (dl.empty()) {
      std::cerr << "STOP: --mode=oracle requires --demand_list_path. The demand "
                   "list is a REGISTERED construction (section 7.2), emitted and "
                   "projected before any cell is computed.\n";
      return 1;
    }
    std::ifstream df(dl);
    if (!df) SpielFatalError("cannot open --demand_list_path");
    std::string c((std::istreambuf_iterator<char>(df)), std::istreambuf_iterator<char>());
    auto p = json::FromString(c);
    if (!p) SpielFatalError("--demand_list_path is not valid JSON");
    std::set<std::pair<int, Action>> uniq;
    int unknown_roots = 0;
    for (const auto& v : p.value().GetArray()) {
      auto o = v.GetObject();
      auto it = root_index.find(o.at("root_id").GetString());
      if (it == root_index.end()) {
        ++unknown_roots;
        continue;
      }
      uniq.insert({it->second, static_cast<Action>(o.at("action").GetInt())});
    }
    if (unknown_roots > 0) {
      std::cerr << "STOP: the demand list names " << unknown_roots
                << " root id(s) absent from the loaded corpora. A silently dropped "
                   "cell would make a registered comparison available-case.\n";
      return 1;
    }
    std::vector<std::pair<int, Action>> reqs(uniq.begin(), uniq.end());
    std::cerr << "  oracle cells: " << reqs.size() << " x "
              << absl::GetFlag(FLAGS_oracle_continuations) << " continuations\n";
    run_oracle_block(reqs, absl::GetFlag(FLAGS_oracle_continuation_offset),
                     absl::GetFlag(FLAGS_oracle_continuations), true, nullptr);
    return 0;
  }

  // =========================================================================
  // MODE: oracle_sanity -- registration 7.3, scoped to the conversion stratum.
  // =========================================================================
  if (mode == "oracle_sanity") {
    std::vector<int> pool;
    for (size_t i = 0; i < roots.size(); ++i)
      if (roots[i].corpus == "conversion" && roots[i].legal_actions.size() >= 2)
        pool.push_back(i);
    if (pool.empty()) {
      std::cerr << "STOP: no conversion-stratum roots loaded.\n";
      return 1;
    }
    auto rng = dune_seed::MakeRng64(
        pwo3::SubsampleSeed(pwo3::kTagConversionSanityRoots));
    std::shuffle(pool.begin(), pool.end(), rng);
    const int nr = std::min<int>(absl::GetFlag(FLAGS_sanity_roots), pool.size());
    pool.resize(nr);
    std::sort(pool.begin(), pool.end());

    std::vector<std::pair<int, Action>> reqs;
    std::vector<int> cell_root;
    for (int i = 0; i < nr; ++i) {
      const int ri = pool[i];
      std::vector<Action> acts = roots[ri].legal_actions;
      const int cap_n = absl::GetFlag(FLAGS_sanity_max_actions);
      if (static_cast<int>(acts.size()) > cap_n) {
        auto arng = dune_seed::MakeRng64(pwo3::SubsampleSeed(
            pwo3::kTagConversionSanityActionsBase + i));
        std::shuffle(acts.begin(), acts.end(), arng);
        acts.resize(cap_n);
        std::sort(acts.begin(), acts.end());
      }
      for (Action a : acts) {
        reqs.push_back({ri, a});
        cell_root.push_back(ri);
      }
    }
    const int n_cont = absl::GetFlag(FLAGS_oracle_continuations);
    std::cerr << "  conversion sanity: " << reqs.size() << " (root,action) cells over "
              << nr << " roots, two blocks of " << n_cont << "\n";
    std::vector<double> b0, b1;
    run_oracle_block(reqs, 0, n_cont, false, &b0);
    run_oracle_block(reqs, n_cont, n_cont, false, &b1);

    // Center within each (root, block), then pool.
    std::map<int, std::pair<double, int>> m0, m1;
    for (size_t i = 0; i < reqs.size(); ++i) {
      m0[cell_root[i]].first += b0[i];
      m0[cell_root[i]].second++;
      m1[cell_root[i]].first += b1[i];
      m1[cell_root[i]].second++;
    }
    std::vector<double> x, y;
    json::Array cells;
    for (size_t i = 0; i < reqs.size(); ++i) {
      const int ri = cell_root[i];
      const double c0 = b0[i] - m0[ri].first / m0[ri].second;
      const double c1 = b1[i] - m1[ri].first / m1[ri].second;
      x.push_back(c0);
      y.push_back(c1);
      json::Object c;
      c["root_id"] = roots[ri].root_id;
      c["stratum"] = roots[ri].stratum;
      c["corpus"] = roots[ri].corpus;
      c["action"] = static_cast<int64_t>(reqs[i].second);
      c["is_conversion"] = pwo3::IsConversion(reqs[i].second);
      c["block0_value"] = b0[i];
      c["block1_value"] = b1[i];
      c["block0_centered"] = c0;
      c["block1_centered"] = c1;
      cells.push_back(c);
    }

    auto rank = [](const std::vector<double>& v) {
      std::vector<size_t> ord(v.size());
      for (size_t i = 0; i < v.size(); ++i) ord[i] = i;
      std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return v[a] < v[b]; });
      std::vector<double> r(v.size());
      size_t i = 0;
      while (i < ord.size()) {
        size_t j = i;
        while (j + 1 < ord.size() && v[ord[j + 1]] == v[ord[i]]) ++j;
        const double avg = (i + j) / 2.0 + 1.0;
        for (size_t k = i; k <= j; ++k) r[ord[k]] = avg;
        i = j + 1;
      }
      return r;
    };
    auto rx = rank(x), ry = rank(y);
    const double mx = std::accumulate(rx.begin(), rx.end(), 0.0) / rx.size();
    const double my = std::accumulate(ry.begin(), ry.end(), 0.0) / ry.size();
    double num = 0, dx = 0, dy = 0;
    for (size_t i = 0; i < rx.size(); ++i) {
      num += (rx[i] - mx) * (ry[i] - my);
      dx += (rx[i] - mx) * (rx[i] - mx);
      dy += (ry[i] - my) * (ry[i] - my);
    }
    const double rho = (dx > 0 && dy > 0) ? num / std::sqrt(dx * dy) : NAN;

    json::Object rep;
    rep["spearman_between_blocks"] = rho;
    rep["threshold"] = 0.5;
    rep["PASS"] = (rho >= 0.5);
    rep["n_cells"] = static_cast<int64_t>(reqs.size());
    rep["n_roots"] = static_cast<int64_t>(m0.size());
    rep["n_continuations_per_block"] = static_cast<int64_t>(n_cont);
    rep["binary_sha256"] = self_sha;
    rep["continuation_policy"] = std::string("branch_a_u2450_raw_temp1.0");
    rep["scope"] = std::string("PWO-3 conversion stratum (registration 7.3)");
    rep["stop_note"] = std::string(
        "Spearman < 0.5 is a registered STOP. The main corpus already passed at "
        "0.7199, so a conversion-stratum failure is information, not noise to "
        "average away.");
    rep["cells"] = cells;
    out << json::ToString(rep, true) << "\n";
    std::cerr << absl::StrFormat("  Spearman = %.4f  -> %s\n", rho,
                                 rho >= 0.5 ? "PASS" : "FAIL (registered STOP)");
    return rho >= 0.5 ? 0 : 2;
  }

  std::cerr << "unknown --mode\n";
  return 1;
}
