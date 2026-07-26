// PWO-2 Phase 1: stratified frozen root corpus.
//
// Implements docs/PWO2_QUALIFICATION_REGISTRATION.md section 3 exactly.
//
// Why a new tool rather than dune_create_corpus: that corpus stratifies by
// "strategic"/"opportunity"/"planner" CATEGORIES whose strategic stratum spans a
// single source episode -- the corpus-probe-artifact class the registered guards
// exist to kill. PWO-2 stratifies by DECISION ROLE and enforces per-stratum
// episode-diversity guards. The replay/reconstruct/history-hash machinery is
// reused; the selection is not.
//
// NEVER uses --legacy_corpus_path semantics: that flag copies stored observation
// arrays verbatim and carries staleness forward. Every observation here is
// produced by, and re-verified against, a live replay under the frozen engine.
//
// Output: a JSON array whose per-root objects are schema-compatible with what
// dune_search_calibration reads via --corpus_path (category/player/round/
// history/observation/legal_actions), plus the PWO-2 tags (root_id, stratum,
// source_arm, source_episode_id, half, history_hash, decision_index, n_legal).

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/random/distributions.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/json.h"
#include <torch/torch.h>

#include "dune_evaluator.h"
#include "dune_network.h"
#include "dune_pwo2_common.h"
#include "dune_search_routing.h"
#include "dune_seed_utils.h"
#include "dune_sha256.h"

ABSL_FLAG(std::string, branch_a_checkpoint,
          "artifacts/branch_a_frozen/branch_a_seed11_model_update_2450.pt",
          "Branch-A u2450 checkpoint (arm 0).");
ABSL_FLAG(std::string, u175_checkpoint,
          "calibration_results_v2/pilot300_search_seed12/ppo_model_update_25.pt",
          "u175 checkpoint (arm 1).");
ABSL_FLAG(std::string, output_path, "data/pwo2_root_corpus.json",
          "Output corpus JSON path.");
ABSL_FLAG(std::string, manifest_path, "data/pwo2_root_corpus_manifest.json",
          "Output manifest JSON path.");
ABSL_FLAG(int, branch_a_base_seed, 777000, "Base seed, Branch-A arm.");
ABSL_FLAG(int, u175_base_seed, 778000, "Base seed, u175 arm.");
ABSL_FLAG(int, episodes_per_arm, 128, "Source episodes per arm (first attempt).");
ABSL_FLAG(int, max_episodes_per_arm, 256,
          "Extra-episode budget ceiling per arm (registration 3.6 step 4).");
ABSL_FLAG(int, half_assign_seed, 20260726, "Seeded coin for calibration/validation halves.");
ABSL_FLAG(int, hidden_dim, 2048, "Hidden dimension.");
ABSL_FLAG(int, num_blocks, 8, "Residual block count.");
ABSL_FLAG(double, logit_cap, 10.0, "Evaluator logit cap.");
ABSL_FLAG(bool, nonlinear_value_head, false, "Versioned nonlinear value head.");
ABSL_FLAG(int, threads, 16, "Worker threads (episodes are independent; results are "
                            "sorted canonically, so this never affects output).");

using namespace open_spiel;

namespace {

// Registered stratum targets (registration 3.3).
struct StratumSpec {
  const char* name;
  DuneDecisionRole role;
  int target;
};
const std::vector<StratumSpec>& Strata() {
  static const std::vector<StratumSpec> kStrata = {
      {"agent_primary", DuneDecisionRole::kAgentPrimary, 64},
      {"agent_continuation", DuneDecisionRole::kAgentContinuation, 48},
      {"purchase", DuneDecisionRole::kPurchase, 48},
      {"combat_intrigue", DuneDecisionRole::kCombatIntrigue, 32},
  };
  return kStrata;
}

constexpr int kNumArms = 2;
const char* ArmName(int arm) { return arm == 0 ? "branch_a" : "u175"; }

// Registered hard guards (registration 3.5).
constexpr int kGuardMinRootsPerStratum = 32;
constexpr int kGuardMinEpisodesPerStratum = 16;
constexpr int kGuardMaxRootsPerEpisodeGlobal = 2;
constexpr int kGuardArmBalanceTolerance = 2;

struct Root {
  int arm = 0;
  int episode_id = 0;
  int decision_index = 0;
  Player player = kInvalidPlayer;
  int round = 0;
  std::string stratum;
  DuneDecisionRole role = DuneDecisionRole::kForcedOrBookkeeping;
  std::vector<Action> history;
  std::vector<Action> legal_actions;
  std::vector<float> observation;
  std::string history_hash;

  // Canonical order: (source_arm, source_episode_id, decision_index).
  bool operator<(const Root& o) const {
    if (arm != o.arm) return arm < o.arm;
    if (episode_id != o.episode_id) return episode_id < o.episode_id;
    return decision_index < o.decision_index;
  }
};

Action SampleFromPrior(const ActionsAndProbs& prior, std::mt19937& rng) {
  if (prior.empty()) return kInvalidAction;
  std::vector<double> w;
  w.reserve(prior.size());
  for (const auto& ap : prior) w.push_back(ap.second);
  std::discrete_distribution<size_t> dist(w.begin(), w.end());
  return prior[dist(rng)].first;
}

std::unique_ptr<State> ReconstructState(const std::shared_ptr<const Game>& game,
                                        const std::vector<Action>& history) {
  auto state = game->NewInitialState();
  for (Action a : history) state->ApplyAction(a);
  return state;
}

std::shared_ptr<SharedDunePolicyValueNetImpl> LoadModel(
    const std::string& path, int64_t obs_size, int64_t action_size,
    torch::Device device) {
  auto model = std::make_shared<SharedDunePolicyValueNetImpl>(
      obs_size, absl::GetFlag(FLAGS_hidden_dim), action_size,
      absl::GetFlag(FLAGS_num_blocks), absl::GetFlag(FLAGS_nonlinear_value_head));
  try {
    torch::load(model, path, device);
  } catch (const std::exception& e) {
    std::cerr << "Failed to load checkpoint " << path << ": " << e.what() << "\n";
    std::exit(1);
  }
  model->to(device);
  model->eval();
  return model;
}

// Registration 3.4: per arm, shuffle the episode ids with a seeded RNG and split
// exactly in half. Guarantees each (source_arm, half) cell gets exactly n/2
// episodes and that no episode's roots can span both halves.
std::map<std::pair<int, int>, std::string> AssignHalves(int episodes_per_arm,
                                                        int half_seed) {
  std::map<std::pair<int, int>, std::string> half;
  for (int arm = 0; arm < kNumArms; ++arm) {
    std::vector<int> ids(episodes_per_arm);
    for (int i = 0; i < episodes_per_arm; ++i) ids[i] = i;
    auto rng = dune_seed::MakeRng64(dune_seed::DeriveSeed(half_seed, arm));
    std::shuffle(ids.begin(), ids.end(), rng);
    for (int i = 0; i < episodes_per_arm; ++i) {
      half[{arm, ids[i]}] = (i < episodes_per_arm / 2) ? "calibration" : "validation";
    }
  }
  return half;
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  at::set_num_threads(1);

  const int num_threads = absl::GetFlag(FLAGS_threads);
  const int half_seed = absl::GetFlag(FLAGS_half_assign_seed);
  const int base_seeds[kNumArms] = {absl::GetFlag(FLAGS_branch_a_base_seed),
                                    absl::GetFlag(FLAGS_u175_base_seed)};
  const std::string ckpts[kNumArms] = {absl::GetFlag(FLAGS_branch_a_checkpoint),
                                       absl::GetFlag(FLAGS_u175_checkpoint)};

  std::cout << "PWO-2 root corpus generator\n"
            << "  Fnv1a64(\"PWO2_SEARCH\") = " << pwo2::kDomainSearch << "\n"
            << "  Fnv1a64(\"PWO2_ORACLE\") = " << pwo2::kDomainOracle << "\n";

  auto game = LoadGame("dune_imperium");
  const int64_t obs_size = game->GetType().provides_information_state_tensor
                               ? game->InformationStateTensorSize()
                               : game->ObservationTensorSize();
  const int64_t action_size = game->NumDistinctActions();
  torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                                     : torch::Device(torch::kCPU);
  std::cout << "  device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";

  std::shared_ptr<SharedDunePolicyValueNetImpl> models[kNumArms];
  std::string model_shas[kNumArms];
  for (int a = 0; a < kNumArms; ++a) {
    models[a] = LoadModel(ckpts[a], obs_size, action_size, device);
    model_shas[a] = pwo2::Sha256File(ckpts[a]);
    std::cout << "  arm " << ArmName(a) << ": " << ckpts[a] << "\n    sha256 "
              << model_shas[a] << "\n";
  }

  // -------------------------------------------------------------------------
  // Collection. One episode is handled entirely by one thread and is seeded by
  // its own game seed, so the collected set is independent of thread count.
  // -------------------------------------------------------------------------
  auto collect = [&](int episodes_per_arm) {
    std::vector<std::vector<Root>> per_thread(num_threads);
    std::atomic<int> next_job{0};
    const int total_jobs = kNumArms * episodes_per_arm;

    auto worker = [&](int tid) {
      std::shared_ptr<algorithms::Evaluator> evals[kNumArms];
      for (int a = 0; a < kNumArms; ++a) {
        evals[a] = std::make_shared<DuneNNEvaluator>(
            models[a], device, static_cast<float>(absl::GetFlag(FLAGS_logit_cap)));
      }
      while (true) {
        int job = next_job.fetch_add(1);
        if (job >= total_jobs) break;
        const int arm = job / episodes_per_arm;
        const int episode_id = job % episodes_per_arm;
        const int game_seed = base_seeds[arm] + episode_id;

        std::mt19937 rng(game_seed);
        auto state = game->NewInitialState();
        int decision_index = 0;
        while (!state->IsTerminal()) {
          ++decision_index;
          if (state->IsChanceNode()) {
            auto outcomes = state->ChanceOutcomes();
            state->ApplyAction(
                SampleAction(outcomes, absl::Uniform(rng, 0.0, 1.0)).first);
            continue;
          }
          if (state->CurrentPlayer() == kSimultaneousPlayerId) {
            std::vector<Action> joint;
            for (int p = 0; p < game->NumPlayers(); ++p) {
              auto acts = state->LegalActions(p);
              std::uniform_int_distribution<int> d(0, acts.size() - 1);
              joint.push_back(acts[d(rng)]);
            }
            state->ApplyActions(joint);
            continue;
          }

          const Player player = state->CurrentPlayer();
          // has_active_session=false: corpus roots are standalone
          // reconstructions with no inherited session, and the qualification
          // runner searches them exactly the same way (registration 3.2).
          const DuneDecisionRole role =
              ClassifyDuneDecisionRole(*state, player, /*has_active_session=*/false);
          auto legal = state->LegalActions();

          for (const auto& sp : Strata()) {
            if (role != sp.role) continue;
            if (legal.size() < 2) break;  // guard G7; classifier implies this
            Root r;
            r.arm = arm;
            r.episode_id = episode_id;
            r.decision_index = decision_index;
            r.player = player;
            const auto* ds =
                dynamic_cast<const dune_imperium::DuneImperiumState*>(state.get());
            r.round = ds ? ds->GetCurrentRound() : 1;
            r.stratum = sp.name;
            r.role = role;
            r.history = state->History();
            r.legal_actions = legal;
            r.observation = state->InformationStateTensor(player);
            r.history_hash = pwo2::HistoryHash(r.history);
            per_thread[tid].push_back(std::move(r));
            break;
          }

          // Four-copy raw-policy self-play at sampling temperature 1.0.
          ActionsAndProbs prior = evals[arm]->Prior(*state);
          state->ApplyAction(SampleFromPrior(prior, rng));
        }
      }
    };

    std::vector<std::thread> ws;
    for (int i = 0; i < num_threads; ++i) ws.emplace_back(worker, i);
    for (auto& w : ws) w.join();

    std::vector<Root> all;
    for (auto& t : per_thread) all.insert(all.end(), t.begin(), t.end());
    std::sort(all.begin(), all.end());  // canonical order
    return all;
  };

  // -------------------------------------------------------------------------
  // Selection (registration 3.6): deterministic, no RNG. Pass 1 takes at most
  // one root per compound episode key to maximize distinct-episode count; pass
  // 2 fills from already-used episodes under the global <= 2 cap.
  // -------------------------------------------------------------------------
  struct SelectResult {
    std::vector<Root> selected;
    std::map<std::string, std::map<int, int>> short_by;  // stratum -> arm -> shortfall
    bool complete = false;
  };

  auto select = [&](const std::vector<Root>& all) {
    SelectResult sr;
    std::set<std::string> seen_hash;
    std::map<std::pair<int, int>, int> per_episode_global;  // (arm,ep) -> count

    for (const auto& sp : Strata()) {
      for (int arm = 0; arm < kNumArms; ++arm) {
        const int cell_target = sp.target / kNumArms;
        int taken = 0;
        std::set<std::pair<int, int>> used_in_cell;

        for (int pass = 0; pass < 2 && taken < cell_target; ++pass) {
          for (const auto& r : all) {
            if (taken >= cell_target) break;
            if (r.arm != arm || r.stratum != sp.name) continue;
            if (seen_hash.count(r.history_hash)) continue;  // G5
            const std::pair<int, int> key{r.arm, r.episode_id};
            const bool already = used_in_cell.count(key) > 0;
            if (pass == 0 && already) continue;
            if (pass == 1 && !already) continue;
            if (per_episode_global[key] >= kGuardMaxRootsPerEpisodeGlobal) continue;  // G3
            seen_hash.insert(r.history_hash);
            per_episode_global[key]++;
            used_in_cell.insert(key);
            sr.selected.push_back(r);
            ++taken;
          }
        }
        if (taken < cell_target) sr.short_by[sp.name][arm] = cell_target - taken;
      }
    }
    sr.complete = sr.short_by.empty();
    std::sort(sr.selected.begin(), sr.selected.end());
    return sr;
  };

  int episodes_per_arm = absl::GetFlag(FLAGS_episodes_per_arm);
  const int max_episodes = absl::GetFlag(FLAGS_max_episodes_per_arm);
  std::cout << "\nGenerating " << episodes_per_arm << " episodes per arm...\n";
  std::vector<Root> all = collect(episodes_per_arm);
  std::cout << "  collected " << all.size() << " eligible roots\n";
  SelectResult sel = select(all);

  if (!sel.complete && episodes_per_arm < max_episodes) {
    std::cout << "\nCell shortfall at " << episodes_per_arm
              << " episodes/arm; extending to " << max_episodes
              << " (registration 3.6 step 4).\n";
    for (const auto& [s, m] : sel.short_by)
      for (const auto& [a, n] : m)
        std::cout << "    short: " << s << " / " << ArmName(a) << " by " << n << "\n";
    episodes_per_arm = max_episodes;
    all = collect(episodes_per_arm);
    std::cout << "  collected " << all.size() << " eligible roots\n";
    sel = select(all);
  }

  if (!sel.complete) {
    std::cerr << "\nSTOP: corpus guards unsatisfiable after the extra-episode budget.\n";
    for (const auto& [s, m] : sel.short_by)
      for (const auto& [a, n] : m)
        std::cerr << "  short: " << s << " / " << ArmName(a) << " by " << n << "\n";
    std::cerr << "Never ship a guard-violating corpus (registration 3.6 / STOP 4).\n";
    return 1;
  }

  auto halves = AssignHalves(episodes_per_arm, half_seed);

  // -------------------------------------------------------------------------
  // Guard verification. Every guard is checked on the FINAL selection and the
  // result table is emitted whether it passes or fails.
  // -------------------------------------------------------------------------
  json::Object guard_table;
  bool all_pass = true;
  std::map<std::pair<int, int>, int> global_ep_count;
  for (const auto& r : sel.selected) global_ep_count[{r.arm, r.episode_id}]++;

  for (const auto& sp : Strata()) {
    int n = 0, per_arm[kNumArms] = {0, 0};
    std::set<std::pair<int, int>> eps;
    std::map<std::pair<int, int>, int> in_stratum;
    for (const auto& r : sel.selected) {
      if (r.stratum != sp.name) continue;
      ++n;
      ++per_arm[r.arm];
      eps.insert({r.arm, r.episode_id});
      in_stratum[{r.arm, r.episode_id}]++;
    }
    int max_in_stratum = 0;
    for (const auto& [k, c] : in_stratum) max_in_stratum = std::max(max_in_stratum, c);

    const bool g1 = n >= kGuardMinRootsPerStratum;
    const bool g2 = static_cast<int>(eps.size()) >= kGuardMinEpisodesPerStratum;
    const bool g4 = std::abs(per_arm[0] - per_arm[1]) <= kGuardArmBalanceTolerance;
    all_pass = all_pass && g1 && g2 && g4 && (n == sp.target);

    json::Object row;
    row["n_roots"] = static_cast<int64_t>(n);
    row["target"] = static_cast<int64_t>(sp.target);
    row["n_roots_branch_a"] = static_cast<int64_t>(per_arm[0]);
    row["n_roots_u175"] = static_cast<int64_t>(per_arm[1]);
    row["n_distinct_compound_episodes"] = static_cast<int64_t>(eps.size());
    row["max_roots_per_episode_within_stratum"] = static_cast<int64_t>(max_in_stratum);
    row["G1_min_32_roots"] = g1;
    row["G2_min_16_episodes"] = g2;
    row["G4_arm_balance_within_2"] = g4;
    guard_table[sp.name] = row;
  }

  int global_max = 0;
  for (const auto& [k, c] : global_ep_count) global_max = std::max(global_max, c);
  const bool g3 = global_max <= kGuardMaxRootsPerEpisodeGlobal;
  all_pass = all_pass && g3;

  // Cross-stratum episode overlap: NOT required to be zero, but recorded.
  std::map<std::pair<int, int>, std::set<std::string>> ep_strata;
  for (const auto& r : sel.selected) ep_strata[{r.arm, r.episode_id}].insert(r.stratum);
  int overlap_eps = 0;
  for (const auto& [k, s] : ep_strata)
    if (s.size() > 1) ++overlap_eps;

  // -------------------------------------------------------------------------
  // G6: every root reconstructs exactly.
  // -------------------------------------------------------------------------
  std::cout << "\nVerifying " << sel.selected.size() << " roots by replay...\n";
  for (const auto& r : sel.selected) {
    auto st = ReconstructState(game, r.history);
    SPIEL_CHECK_EQ(st->CurrentPlayer(), r.player);
    const auto* ds = dynamic_cast<const dune_imperium::DuneImperiumState*>(st.get());
    SPIEL_CHECK_EQ(ds ? ds->GetCurrentRound() : 1, r.round);
    SPIEL_CHECK_TRUE(st->LegalActions() == r.legal_actions);
    SPIEL_CHECK_GE(r.legal_actions.size(), 2u);  // G7
    SPIEL_CHECK_TRUE(ClassifyDuneDecisionRole(*st, r.player, false) == r.role);
    auto obs = st->InformationStateTensor(r.player);
    SPIEL_CHECK_EQ(obs.size(), r.observation.size());
    for (size_t i = 0; i < obs.size(); ++i)
      SPIEL_CHECK_FLOAT_NEAR(obs[i], r.observation[i], 1e-5);
    SPIEL_CHECK_EQ(pwo2::HistoryHash(r.history), r.history_hash);
  }
  std::cout << "  all roots replay-verified (G6, G7).\n";

  // -------------------------------------------------------------------------
  // Live-50 registered 96-root subset (registration 3.7), fixed here, BEFORE
  // any search runs.
  // -------------------------------------------------------------------------
  std::set<std::string> live50;
  for (const auto& r : sel.selected)
    if (r.stratum == "agent_primary") live50.insert(r.history_hash);
  const std::map<std::string, std::pair<int, int>> diag = {  // stratum -> (index, k)
      {"agent_continuation", {1, 11}}, {"purchase", {2, 11}}, {"combat_intrigue", {3, 10}}};
  for (const auto& [name, ik] : diag) {
    std::vector<std::string> pool;
    for (const auto& r : sel.selected)
      if (r.stratum == name) pool.push_back(r.history_hash);
    auto rng = dune_seed::MakeRng64(dune_seed::DeriveSeed(half_seed, ik.first));
    std::shuffle(pool.begin(), pool.end(), rng);
    for (int i = 0; i < ik.second && i < static_cast<int>(pool.size()); ++i)
      live50.insert(pool[i]);
  }
  std::cout << "  live-50 registered subset: " << live50.size() << " roots\n";

  // -------------------------------------------------------------------------
  // Emit.
  // -------------------------------------------------------------------------
  json::Array corpus;
  for (const auto& r : sel.selected) {
    json::Object o;
    // Schema-compatible with dune_search_calibration --corpus_path.
    o["category"] = r.stratum;
    o["player"] = static_cast<int64_t>(r.player);
    o["round"] = static_cast<int64_t>(r.round);
    json::Array h;
    for (Action a : r.history) h.push_back(static_cast<int64_t>(a));
    o["history"] = h;
    json::Array obs;
    for (float v : r.observation) obs.push_back(v);
    o["observation"] = obs;
    json::Array la;
    for (Action a : r.legal_actions) la.push_back(static_cast<int64_t>(a));
    o["legal_actions"] = la;
    // PWO-2 tags.
    o["root_id"] = r.history_hash;
    o["history_hash"] = r.history_hash;
    o["stratum"] = r.stratum;
    o["role"] = pwo2::RoleName(r.role);
    o["source_arm"] = std::string(ArmName(r.arm));
    o["source_episode_id"] = static_cast<int64_t>(r.episode_id);
    o["source_game_seed"] = static_cast<int64_t>(base_seeds[r.arm] + r.episode_id);
    o["half"] = halves.at({r.arm, r.episode_id});
    o["decision_index"] = static_cast<int64_t>(r.decision_index);
    o["n_legal"] = static_cast<int64_t>(r.legal_actions.size());
    o["in_live50_subset"] = live50.count(r.history_hash) > 0;
    o["corpus_schema_version"] = std::string("pwo2_v1");
    corpus.push_back(o);
  }

  std::filesystem::create_directories(
      std::filesystem::path(absl::GetFlag(FLAGS_output_path)).parent_path());
  {
    std::ofstream out(absl::GetFlag(FLAGS_output_path));
    if (!out) {
      std::cerr << "Failed to open output path.\n";
      return 1;
    }
    out << json::ToString(corpus, true) << "\n";
  }

  json::Object man;
  man["tool"] = std::string("dune_pwo2_root_corpus");
  man["registration"] = std::string("docs/PWO2_QUALIFICATION_REGISTRATION.md");
  man["binary_sha256"] = pwo2::Sha256File(argv[0]);
  man["episodes_per_arm"] = static_cast<int64_t>(episodes_per_arm);
  man["branch_a_base_seed"] = static_cast<int64_t>(base_seeds[0]);
  man["u175_base_seed"] = static_cast<int64_t>(base_seeds[1]);
  man["half_assign_seed"] = static_cast<int64_t>(half_seed);
  man["sampling_temperature"] = 1.0;
  man["logit_cap"] = absl::GetFlag(FLAGS_logit_cap);
  man["hidden_dim"] = static_cast<int64_t>(absl::GetFlag(FLAGS_hidden_dim));
  man["num_blocks"] = static_cast<int64_t>(absl::GetFlag(FLAGS_num_blocks));
  man["nonlinear_value_head"] = absl::GetFlag(FLAGS_nonlinear_value_head);
  man["threads"] = static_cast<int64_t>(num_threads);
  man["n_roots"] = static_cast<int64_t>(sel.selected.size());
  man["n_eligible_roots_collected"] = static_cast<int64_t>(all.size());
  man["guard_table"] = guard_table;
  man["G3_max_roots_per_compound_episode_global"] = static_cast<int64_t>(global_max);
  man["G3_pass"] = g3;
  man["G5_no_duplicate_history_hash"] = true;  // enforced during selection
  man["ALL_GUARDS_PASS"] = all_pass;
  man["cross_stratum_episode_overlap_count"] = static_cast<int64_t>(overlap_eps);
  man["cross_stratum_overlap_note"] =
      std::string("Episode-disjointness BETWEEN strata is not required; recorded only.");
  man["live50_subset_size"] = static_cast<int64_t>(live50.size());
  man["domain_tag_PWO2_SEARCH"] = static_cast<int64_t>(pwo2::kDomainSearch);
  man["domain_tag_PWO2_ORACLE"] = static_cast<int64_t>(pwo2::kDomainOracle);
  json::Object shas;
  shas["branch_a"] = model_shas[0];
  shas["u175"] = model_shas[1];
  man["model_sha256"] = shas;
  json::Object per_half;
  for (const char* hh : {"calibration", "validation"}) {
    int c = 0;
    for (const auto& r : sel.selected)
      if (halves.at({r.arm, r.episode_id}) == hh) ++c;
    per_half[hh] = static_cast<int64_t>(c);
  }
  man["roots_per_half"] = per_half;
  {
    std::ofstream out(absl::GetFlag(FLAGS_manifest_path));
    out << json::ToString(man, true) << "\n";
  }

  std::cout << "\nGuard table:\n";
  for (const auto& sp : Strata()) {
    auto row = guard_table[sp.name].GetObject();
    std::cout << absl::StrFormat(
        "  %-20s n=%3d/%3d  arms %3d/%3d  episodes=%3d  G1=%s G2=%s G4=%s\n",
        sp.name, row.at("n_roots").GetInt(), row.at("target").GetInt(),
        row.at("n_roots_branch_a").GetInt(), row.at("n_roots_u175").GetInt(),
        row.at("n_distinct_compound_episodes").GetInt(),
        row.at("G1_min_32_roots").GetBool() ? "PASS" : "FAIL",
        row.at("G2_min_16_episodes").GetBool() ? "PASS" : "FAIL",
        row.at("G4_arm_balance_within_2").GetBool() ? "PASS" : "FAIL");
  }
  std::cout << absl::StrFormat("  G3 max roots per compound episode (global) = %d  %s\n",
                               global_max, g3 ? "PASS" : "FAIL");
  std::cout << "  ALL_GUARDS_PASS = " << (all_pass ? "true" : "false") << "\n";
  std::cout << "\nWrote " << absl::GetFlag(FLAGS_output_path) << " and "
            << absl::GetFlag(FLAGS_manifest_path) << "\n";
  return all_pass ? 0 : 1;
}
