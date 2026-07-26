// PWO-2 Phases 2+3: qualification runner and rollout oracle.
//
// Implements docs/PWO2_QUALIFICATION_REGISTRATION.md section 4.
//
// Modes:
//   --mode=search         candidates x tiers x seeds over the corpus -> JSONL
//   --mode=oracle         oracle values for (root, action) pairs      -> JSONL
//   --mode=oracle_sanity  two-block Spearman sanity check             -> JSON
//
// ---------------------------------------------------------------------------
// TWO DESIGN DECISIONS THAT ARE NOT COSMETIC
// ---------------------------------------------------------------------------
//
// 1. PER-ROOT FRESH FULL SEARCH, NOT A SESSION.
//    Registration 2/4 specify "per-root fresh full search by protocol". This is
//    load-bearing, not stylistic: DuneSearchSession routes kPurchase and
//    kCombatIntrigue roots through the SHORT-WINDOW budget
//    (dune_search_session.cc:344, max_sims = purchase_combat_budget - ...), so a
//    session-driven runner would silently give every purchase/combat root 16
//    simulations at ALL THREE of the 64/200/800 tiers. Driving
//    DunePUCTISMCTSBot::RunSearch directly makes each tier mean what it says at
//    every root. Consequence: purchase_combat_budget is a SESSION knob and is
//    therefore inert here (pinned but unconsulted); it stays live and pinned at
//    16 for the Phase 4 baselines, which do use sessions. Same for
//    fixed_continuation_reserve and live_continuation_reserve_seconds.
//
// 2. SELECTION IS ARGMAX, INCLUDING ON THE FALLBACK PATH.
//    DunePUCTISMCTSBot::Step() SAMPLES from result.policy
//    (dune_puct_is_mcts.cc:1076). Under the registered --temperature=0.0 the
//    searched policy is one-hot, so sampling is a no-op there -- but on a
//    STARVED root result.policy is the raw network prior, which is not one-hot,
//    and Step() would sample it. Sampling a raw prior under a temperature-0 pin
//    is exactly the documented below-parity class (temp-1.0 sampling at temp 0,
//    2026-07-12; starved paths must degrade to raw-prior ARGMAX, 2026-07-20).
//    This runner therefore selects by deterministic argmax over result.policy
//    with a lowest-action-id tie-break, and records used_fallback /
//    fallback_reason on every row so starvation stays visible rather than
//    becoming noise in the verdict.

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
#include "dune_search_routing.h"
#include "dune_search_session.h"  // SampleActionFromPrior(prior, r_val)
#include "dune_seed_utils.h"
#include "dune_sha256.h"

ABSL_FLAG(std::string, mode, "search", "search | oracle | oracle_sanity");
ABSL_FLAG(std::string, corpus_path, "data/pwo2_root_corpus.json", "Root corpus.");
ABSL_FLAG(std::string, output_path, "", "Output JSONL/JSON path.");
ABSL_FLAG(std::string, branch_a_checkpoint,
          "artifacts/branch_a_frozen/branch_a_seed11_model_update_2450.pt", "");
ABSL_FLAG(std::string, u175_checkpoint,
          "calibration_results_v2/pilot300_search_seed12/ppo_model_update_25.pt", "");
ABSL_FLAG(int, hidden_dim, 2048, "");
ABSL_FLAG(int, num_blocks, 8, "");
ABSL_FLAG(bool, nonlinear_value_head, false, "");
ABSL_FLAG(int, threads, 1, "Worker threads.");

// ---- The registered controller pin (registration section 2). Every flag is
// passed explicitly by launch.sh; no default here defines a measurement. -----
ABSL_FLAG(int, opponent_mode, 1, "0=kMaxN, 1=policy sampling. Pin: 1.");
ABSL_FLAG(double, simulated_opponent_temperature, 1.0, "Pin: 1.0");
ABSL_FLAG(double, temperature, 0.0, "Pin: 0.0 (greedy)");
ABSL_FLAG(double, puct_c, 0.3, "Pin: 0.3 (calibration heritage default is 1.0)");
ABSL_FLAG(double, root_prior_temperature, 1.0, "Pin: 1.0");
ABSL_FLAG(double, utility_divisor, 4.0, "Pin: 4.0");
ABSL_FLAG(int, max_world_samples, -1, "Pin: -1");
ABSL_FLAG(double, dirichlet_epsilon, 0.0, "Pin: 0.0 (root noise off)");
ABSL_FLAG(bool, check_strategic_state, false, "Pin: false");
ABSL_FLAG(double, candidate_logit_cap, 10.0, "Pin: 10.0");
ABSL_FLAG(int, max_nodes, 200000, "Pin: 200000 global (registration 2.2).");
ABSL_FLAG(int, min_visit_threshold, 2, "");

// ---- Grid selection -------------------------------------------------------
ABSL_FLAG(std::string, tiers, "fixed_64,fixed_200,fixed_800,live_50", "");
ABSL_FLAG(std::string, candidates, "branch_a,u175", "");
ABSL_FLAG(std::string, search_seeds, "1301,1302", "");
ABSL_FLAG(std::string, strata, "", "Restrict to these strata (empty = all).");
ABSL_FLAG(int, root_limit, 0, "Cap roots (canonical order) for pilots. 0 = all.");
ABSL_FLAG(int, live_max_simulations, 10000,
          "Non-binding live sim ceiling. The benchmark default of 50 would turn "
          "'50 seconds' into '50 simulations'.");
ABSL_FLAG(double, live_deadline_ms, 50000.0, "Live tier wall-clock deadline.");

// ---- Oracle ---------------------------------------------------------------
ABSL_FLAG(std::string, from_results, "",
          "Extract distinct (root_id, chosen_action) pairs from this results JSONL.");
ABSL_FLAG(int, oracle_continuations, 128, "Registered: 128.");
ABSL_FLAG(int, oracle_continuation_offset, 0, "Block offset (sanity uses 0 and 128).");
ABSL_FLAG(int, oracle_max_actions_per_root, 0,
          "0 = only actions named by --from_results; >0 = that many legal actions per root.");
ABSL_FLAG(bool, emit_per_continuation, false,
          "Emit the per-continuation returns digest (thread-reproducibility gate).");
ABSL_FLAG(int, sanity_roots_per_stratum, 4, "");
ABSL_FLAG(int, sanity_max_actions, 12, "");
ABSL_FLAG(int, sanity_seed, 20260726, "");

using namespace open_spiel;

namespace {

struct Root {
  std::string root_id;      // == history_hash
  std::string stratum;
  std::string source_arm;
  std::string half;
  int source_episode_id = 0;
  int round = 0;
  Player player = kInvalidPlayer;
  std::vector<Action> history;
  std::vector<Action> legal_actions;
  bool in_live50 = false;
};

struct Tier {
  std::string name;
  int max_sims;
  double time_budget_ms;
  bool live;
};

Tier ResolveTier(const std::string& name) {
  const double kInf = std::numeric_limits<double>::infinity();
  if (name == "fixed_64") return {"fixed_64", 64, kInf, false};
  if (name == "fixed_200") return {"fixed_200", 200, kInf, false};
  if (name == "fixed_800") return {"fixed_800", 800, kInf, false};
  if (name == "live_50")
    return {"live_50", absl::GetFlag(FLAGS_live_max_simulations),
            absl::GetFlag(FLAGS_live_deadline_ms), true};
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

// The registered controller pin, assembled in one place.
DuneSearchConfig MakePinnedConfig(const Tier& tier, uint64_t search_rng_seed) {
  DuneSearchConfig c;
  c.max_simulations = tier.max_sims;
  c.relative_time_budget_ms = tier.time_budget_ms;
  c.max_nodes = absl::GetFlag(FLAGS_max_nodes);
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
  c.seed = search_rng_seed;
  // Session-only knobs. Unconsulted here (see header note 1); set to their
  // registered no-op values so a manifest diff never suggests otherwise.
  c.fixed_continuation_reserve = 0;
  c.purchase_combat_budget = 16;
  c.live_continuation_reserve_seconds = 0.0;
  c.fixed_session_limit = tier.max_sims;
  return c;
}

// Deterministic argmax with lowest-action-id tie-break. See header note 2.
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

// One raw-policy playout to terminal. All four seats play `evaluator`'s prior
// sampled at temperature 1.0. No critic bootstrap, ever.
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

std::vector<Root> LoadCorpus(const std::string& path,
                             const std::shared_ptr<const Game>& game) {
  std::ifstream f(path);
  if (!f) SpielFatalError("cannot open corpus: " + path);
  std::string content((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
  auto parsed = json::FromString(content);
  if (!parsed) SpielFatalError("corpus is not valid JSON");
  std::vector<Root> roots;
  for (const auto& v : parsed.value().GetArray()) {
    auto o = v.GetObject();
    Root r;
    r.root_id = o.at("root_id").GetString();
    r.stratum = o.at("stratum").GetString();
    r.source_arm = o.at("source_arm").GetString();
    r.half = o.at("half").GetString();
    r.source_episode_id = static_cast<int>(o.at("source_episode_id").GetInt());
    r.round = static_cast<int>(o.at("round").GetInt());
    r.player = static_cast<Player>(o.at("player").GetInt());
    for (const auto& a : o.at("history").GetArray())
      r.history.push_back(static_cast<Action>(a.GetInt()));
    for (const auto& a : o.at("legal_actions").GetArray())
      r.legal_actions.push_back(static_cast<Action>(a.GetInt()));
    r.in_live50 = o.find("in_live50_subset") != o.end() &&
                  o.at("in_live50_subset").GetBool();
    // Fail closed: the corpus must still replay under the frozen engine.
    SPIEL_CHECK_EQ(pwo2::HistoryHash(r.history), r.root_id);
    roots.push_back(std::move(r));
  }
  (void)game;
  return roots;
}

std::string FmtExact(double v) { return absl::StrFormat("%.17g", v); }

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
  const std::string ck_b = absl::GetFlag(FLAGS_u175_checkpoint);
  auto model_a = LoadModel(ck_a, obs_size, act_size, device);
  auto model_b = LoadModel(ck_b, obs_size, act_size, device);
  const std::string sha_a = pwo2::Sha256File(ck_a);
  const std::string sha_b = pwo2::Sha256File(ck_b);
  const float cap = static_cast<float>(absl::GetFlag(FLAGS_candidate_logit_cap));

  std::cerr << "dune_pwo2_qualify mode=" << mode << " threads=" << num_threads
            << " device=" << (device.is_cuda() ? "CUDA" : "CPU")
            << "\n  binary sha256 " << self_sha
            << "\n  branch_a " << sha_a << "\n  u175     " << sha_b
            << "\n  Fnv1a64(PWO2_SEARCH)=" << pwo2::kDomainSearch
            << " Fnv1a64(PWO2_ORACLE)=" << pwo2::kDomainOracle << "\n";

  auto roots = LoadCorpus(absl::GetFlag(FLAGS_corpus_path), game);
  const auto keep_strata = SplitCsv(absl::GetFlag(FLAGS_strata));
  if (!keep_strata.empty()) {
    std::vector<Root> f;
    for (auto& r : roots)
      if (std::find(keep_strata.begin(), keep_strata.end(), r.stratum) != keep_strata.end())
        f.push_back(r);
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
      std::string cand;
      Tier tier;
      int seed;
    };
    std::vector<Job> jobs;
    for (const auto& tname : SplitCsv(absl::GetFlag(FLAGS_tiers))) {
      Tier t = ResolveTier(tname);
      for (size_t i = 0; i < roots.size(); ++i) {
        // live_50 runs only on the registered 96-root subset (registration 3.7).
        if (t.live && !roots[i].in_live50) continue;
        for (const auto& c : SplitCsv(absl::GetFlag(FLAGS_candidates)))
          for (const auto& s : SplitCsv(absl::GetFlag(FLAGS_search_seeds)))
            jobs.push_back({static_cast<int>(i), c, t, std::stoi(s)});
      }
    }
    std::cerr << "  jobs: " << jobs.size() << "\n";

    std::atomic<int> next{0};
    std::atomic<int> done{0};
    auto worker = [&]() {
      auto ev_a = std::make_shared<DuneNNEvaluator>(model_a, device, cap);
      auto ev_b = std::make_shared<DuneNNEvaluator>(model_b, device, cap);
      while (true) {
        int ji = next.fetch_add(1);
        if (ji >= static_cast<int>(jobs.size())) break;
        const Job& j = jobs[ji];
        const Root& r = roots[j.root];
        auto state = Reconstruct(game, r.history);
        SPIEL_CHECK_EQ(state->CurrentPlayer(), r.player);

        std::shared_ptr<algorithms::Evaluator> ev =
            (j.cand == "branch_a") ? std::static_pointer_cast<algorithms::Evaluator>(ev_a)
                                   : std::static_pointer_cast<algorithms::Evaluator>(ev_b);

        const uint64_t rng_seed = pwo2::SearchRngSeed(j.seed, r.root_id);
        DuneSearchConfig cfg = MakePinnedConfig(j.tier, rng_seed);
        // Fresh bot per (root, candidate, tier, seed): no inherited tree, no
        // session budget. See header note 1.
        DunePUCTISMCTSBot bot(cfg, ev);
        DuneSearchResult res =
            bot.RunSearch(*state, j.tier.max_sims, j.tier.time_budget_ms, 0);
        const Action chosen = ArgmaxPolicy(res.policy);  // header note 2
        SearchDiagnostics d = bot.GetRootDiagnostics(*state, cfg.min_visit_threshold, chosen);

        json::Object o;
        o["root_id"] = r.root_id;
        o["stratum"] = r.stratum;
        o["source_arm"] = r.source_arm;
        o["source_episode_id"] = static_cast<int64_t>(r.source_episode_id);
        o["half"] = r.half;
        o["round"] = static_cast<int64_t>(r.round);
        o["acting_player"] = static_cast<int64_t>(r.player);
        o["n_legal"] = static_cast<int64_t>(r.legal_actions.size());
        o["candidate"] = j.cand;
        o["tier"] = j.tier.name;
        o["seed"] = static_cast<int64_t>(j.seed);
        o["search_rng_seed"] = static_cast<int64_t>(rng_seed);
        o["chosen_action"] = static_cast<int64_t>(chosen);
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
        o["chosen_action_raw_prior_rank"] = static_cast<int64_t>(d.chosen_action_raw_prior_rank);
        o["action_changed_vs_raw_argmax"] = d.action_changed_vs_raw_argmax;
        o["configured_max_simulations"] = static_cast<int64_t>(j.tier.max_sims);
        o["configured_max_nodes"] = static_cast<int64_t>(cfg.max_nodes);
        o["configured_time_budget_ms"] =
            std::isinf(j.tier.time_budget_ms) ? -1.0 : j.tier.time_budget_ms;
        o["binary_sha256"] = self_sha;
        o["model_sha256"] = (j.cand == "branch_a") ? sha_a : sha_b;
        json::Array visits;
        for (int v : d.visit_counts) visits.push_back(static_cast<int64_t>(v));
        o["visit_counts"] = visits;
        json::Array acts;
        for (Action a : d.actions) acts.push_back(static_cast<int64_t>(a));
        o["root_actions"] = acts;

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
    std::cerr << "\n  search done\n";
    return 0;
  }

  // =========================================================================
  // MODE: oracle  (and the request-building shared by oracle_sanity)
  // =========================================================================
  // The continuation policy is ALWAYS Branch-A u2450 raw at temperature 1.0,
  // for every root and every candidate's action (registration 4.4).
  auto run_oracle_block = [&](const std::vector<std::pair<int, Action>>& reqs,
                              int offset, int n_cont,
                              bool emit_rows, bool emit_digest,
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
          const int idx = offset + k;
          const uint64_t s = pwo2::OracleContinuationSeed(r.root_id, idx);
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
          o["action"] = static_cast<int64_t>(a);
          o["value"] = value;
          o["n_continuations"] = static_cast<int64_t>(n_cont);
          o["continuation_offset"] = static_cast<int64_t>(offset);
          o["continuation_policy"] = std::string("branch_a_u2450_raw_temp1.0");
          o["binary_sha256"] = self_sha;
          o["model_sha256"] = sha_a;
          if (emit_digest) {
            SHA256 h;
            for (double v : per) h.Update(FmtExact(v) + ";");
            o["per_continuation_digest"] = h.Final();
          }
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

  std::map<std::string, int> root_index;
  for (size_t i = 0; i < roots.size(); ++i) root_index[roots[i].root_id] = i;

  if (mode == "oracle") {
    std::set<std::pair<int, Action>> uniq;
    const std::string from = absl::GetFlag(FLAGS_from_results);
    if (!from.empty()) {
      std::ifstream rf(from);
      if (!rf) {
        std::cerr << "cannot open --from_results\n";
        return 1;
      }
      std::string line;
      while (std::getline(rf, line)) {
        if (line.empty()) continue;
        auto p = json::FromString(line);
        if (!p) continue;
        auto o = p.value().GetObject();
        auto it = root_index.find(o.at("root_id").GetString());
        if (it == root_index.end()) continue;
        uniq.insert({it->second, static_cast<Action>(o.at("chosen_action").GetInt())});
      }
    }
    const int per_root = absl::GetFlag(FLAGS_oracle_max_actions_per_root);
    if (per_root > 0) {
      for (size_t i = 0; i < roots.size(); ++i)
        for (int k = 0; k < per_root && k < static_cast<int>(roots[i].legal_actions.size()); ++k)
          uniq.insert({static_cast<int>(i), roots[i].legal_actions[k]});
    }
    std::vector<std::pair<int, Action>> reqs(uniq.begin(), uniq.end());
    std::cerr << "  oracle cells: " << reqs.size() << " x "
              << absl::GetFlag(FLAGS_oracle_continuations) << " continuations\n";
    run_oracle_block(reqs, absl::GetFlag(FLAGS_oracle_continuation_offset),
                     absl::GetFlag(FLAGS_oracle_continuations),
                     /*emit_rows=*/true,
                     absl::GetFlag(FLAGS_emit_per_continuation), nullptr);
    return 0;
  }

  // =========================================================================
  // MODE: oracle_sanity  (registration 4.6)
  // =========================================================================
  if (mode == "oracle_sanity") {
    const int seed = absl::GetFlag(FLAGS_sanity_seed);
    const std::vector<std::string> strata = {"agent_primary", "agent_continuation",
                                             "purchase", "combat_intrigue"};
    std::vector<std::pair<int, Action>> reqs;
    std::vector<int> cell_root;
    for (size_t si = 0; si < strata.size(); ++si) {
      std::vector<int> pool;
      for (size_t i = 0; i < roots.size(); ++i)
        if (roots[i].stratum == strata[si] && roots[i].legal_actions.size() >= 2)
          pool.push_back(i);
      auto rng = dune_seed::MakeRng64(dune_seed::DeriveSeed(seed, 100 + si));
      std::shuffle(pool.begin(), pool.end(), rng);
      const int nr = std::min<int>(absl::GetFlag(FLAGS_sanity_roots_per_stratum), pool.size());
      for (int i = 0; i < nr; ++i) {
        const int ri = pool[i];
        std::vector<Action> acts = roots[ri].legal_actions;
        const int cap_n = absl::GetFlag(FLAGS_sanity_max_actions);
        if (static_cast<int>(acts.size()) > cap_n) {
          auto arng = dune_seed::MakeRng64(dune_seed::DeriveSeed(seed, 200 + si, i));
          std::shuffle(acts.begin(), acts.end(), arng);
          acts.resize(cap_n);
          std::sort(acts.begin(), acts.end());
        }
        for (Action a : acts) {
          reqs.push_back({ri, a});
          cell_root.push_back(ri);
        }
      }
    }
    const int n_cont = absl::GetFlag(FLAGS_oracle_continuations);
    std::cerr << "  sanity: " << reqs.size() << " (root,action) cells, two blocks of "
              << n_cont << "\n";
    std::vector<double> b0, b1;
    run_oracle_block(reqs, 0, n_cont, false, false, &b0);
    run_oracle_block(reqs, n_cont, n_cont, false, false, &b1);

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
      c["action"] = static_cast<int64_t>(reqs[i].second);
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
    rep["cells"] = cells;
    out << json::ToString(rep, true) << "\n";
    std::cerr << absl::StrFormat("  Spearman = %.4f  -> %s\n", rho,
                                 rho >= 0.5 ? "PASS" : "FAIL (STOP: oracle noise "
                                                       "swamps signal at 128)");
    return rho >= 0.5 ? 0 : 2;
  }

  std::cerr << "unknown --mode\n";
  return 1;
}
