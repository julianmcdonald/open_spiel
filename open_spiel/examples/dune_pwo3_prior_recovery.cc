// PWO-3 Amendment 2 -- sidecar precision recovery.
//
// Protocol: docs/PWO3_AMENDMENT_2_PRECISION_RECOVERY.md section 3
// (the recovery instrument) and section 4 (the battery this feeds).
//
// ---------------------------------------------------------------------------
// WHAT THIS TOOL IS FOR
// ---------------------------------------------------------------------------
// open_spiel/utils/json.cc:302 emits doubles with std::to_string -- a fixed
// six-decimal %f. Any raw prior below 5e-7 therefore serialized as exactly
// "0.0" in every PWO-3 grid row. The frozen guard in
// scripts/eval/pwo3_common.py (kl_visits_vs_prior) hard-fails on a VISITED
// action carrying prior 0.0, because the evaluator's capped softmax cannot
// produce an exact zero in memory. That premise is true in memory and false in
// the serialized data. This tool restores it by recomputing the priors at full
// double precision and writing them at round-trip precision -- a SIDECAR. It
// changes no guard, no estimand and no stored row.
//
// ---------------------------------------------------------------------------
// WHY THIS IS A NEW BINARY AND NOT A MODE ON dune_pwo3_teacher_audit
// ---------------------------------------------------------------------------
// dune_pwo3_teacher_audit is the binary of record for every PWO-3 measurement
// (sha256 02874e8f...). Adding a mode would relink it, and every committed
// measurement would then be attributed to a binary that no longer exists. This
// tool is built as its own target and the audit binary is asserted unchanged
// after the build. The shared emission chain -- corpus loading, reconstruction,
// PriorAndEvaluate, the expansion writes, FilterAndNormalizeRawPriors -- is
// reached through the SAME code in both, via dune_pwo3_root_loader.h and the
// dune_search library.
//
// ---------------------------------------------------------------------------
// THE REPLAYED CHAIN (amendment section 3), with ZERO simulations
// ---------------------------------------------------------------------------
//  1. LoadCorpusInto + Reconstruct                     dune_pwo3_root_loader.h
//  2. DunePUCTISMCTSBot::RunSearch(state, 0, inf, 0)   dune_puct_is_mcts.cc:823
//     -> LookupOrCreateNode + InitializePriorsAndValue (:268)
//        -> DuneNNEvaluator::PriorAndEvaluate          dune_evaluator.h:156
//           (AutocastGuard FP16 on CUDA, {num_players, obs} stacked batch,
//            FP32 cast, CPU DOUBLE softmax with max subtraction)
//        -> child.raw_prior = priors[i].second         dune_puct_is_mcts.cc:320
//           (after the :285 pre-init of every legal action to 0.0)
//  3. GetRootDiagnostics -> FilterAndNormalizeRawPriors (:430), read back as
//     diag.raw_priors (:1205-1207) -- byte-for-byte the vector the grid wrote.
//
// `Prior()` is NOT used. It runs a single-row batch, which is different
// numerics, and it feeds the "priors" branch of GetRootDiagnostics -- the branch
// ZERO PWO-3 rows took (every row carries raw_prior_source == "raw_priors").
// The branch actually taken is emitted per root so the battery can prove it.
//
// Zero simulations is sound because root raw priors are IMMUTABLE after
// expansion: nothing in the simulation loop writes child_info[a].raw_prior at
// the root. That is not assumed here -- dune_pwo3_immutability_test proves it
// by running a real several-hundred-simulation search and comparing the
// post-search emission bitwise against the zero-simulation snapshot.
//
// DETERMINISM: single-threaded, roots processed in sorted root_id order, and NO
// timestamp in the data file. Wall clock goes to stderr (the run log). Two
// fresh-process runs must produce byte-identical output; that is battery
// check 1.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/json.h"

#include <ATen/Version.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <torch/torch.h>
#include <torch/version.h>

#include "dune_evaluator.h"
#include "dune_network.h"
#include "dune_puct_is_mcts.h"
#include "dune_pwo2_common.h"
#include "dune_pwo3_common.h"
#include "dune_pwo3_root_loader.h"

ABSL_FLAG(std::string, main_corpus, "data/pwo2_root_corpus.json", "");
ABSL_FLAG(std::string, conversion_corpus, "", "PWO-3 conversion stratum.");
ABSL_FLAG(std::string, corpora, "main,conversion",
          "Which corpora to recover. The SM stratum carries no lambda row "
          "(registration 5.3) and is not part of the SM-free lambda set.");
ABSL_FLAG(std::string, output_path, "", "Output JSONL path.");
ABSL_FLAG(std::string, branch_a_checkpoint,
          "artifacts/branch_a_frozen/branch_a_seed11_model_update_2450.pt",
          "The teacher. The ONLY model ever searched in this WO.");
ABSL_FLAG(int, hidden_dim, 2048, "");
ABSL_FLAG(int, num_blocks, 8, "");
ABSL_FLAG(bool, nonlinear_value_head, false, "");
ABSL_FLAG(int, threads, 1, "Pinned to 1. Any other value is refused.");
ABSL_FLAG(std::string, engine_tree_hash, "",
          "REQUIRED. `git rev-parse HEAD:games/dune_imperium`, recorded in the "
          "provenance block. The recovery gate re-derives and compares it.");

// ---- The inherited controller pin (registration section 2), passed explicitly
// exactly as the grid launcher passes it. Only --candidate_logit_cap reaches the
// raw prior; the rest are pinned so the config is identical rather than merely
// equivalent, and so a manifest diff can never suggest otherwise. -------------
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
ABSL_FLAG(double, covered_prior_threshold, 0.50, "Pin: 0.50");
ABSL_FLAG(double, stability_checkpoint_fraction, 0.5, "Pin: 0.5");
// The tier whose config shape is reproduced. The recovery runs ZERO simulations
// at every tier, and the raw prior is written once at expansion, so the tier
// cannot change the answer -- but the seed-1301 fixed_800 rows are the lambda
// rows (registration 12.1), so that is the shape recorded.
ABSL_FLAG(int, recovery_reference_max_simulations, 800, "fixed_800 shape.");
ABSL_FLAG(int, recovery_reference_seed, 1301,
          "The lambda-row search seed (registration 12.1). Drawn into the "
          "config's RNG stream, which no raw prior reads.");

using namespace open_spiel;

namespace {

// The audit tool's exact-decimal helper (dune_pwo3_teacher_audit.cc:306):
// %.17g is max_digits10 for IEEE double, so parsing the string reproduces the
// bit pattern.
std::string FmtExact(double v) { return absl::StrFormat("%.17g", v); }

// The hex-float form. Redundant with %.17g by construction, and kept anyway:
// it is the one representation whose round-trip does not depend on a decimal
// parser agreeing with a decimal printer.
std::string FmtHex(double v) { return absl::StrFormat("%a", v); }

// The LEGACY six-decimal projection, produced by CALLING open_spiel's own
// json.cc rather than by re-implementing its formatting. json::ToString on a
// double Value is exactly what wrote every stored raw_prior_vector entry
// (json.cc:301-302, std::to_string). Re-implementing "%.6f" in C++ or in Python
// would be a second instrument for the very quantity under audit.
std::string LegacyProjection(double v) { return json::ToString(json::Value(v)); }

std::vector<std::string> SplitCsv(const std::string& s) {
  std::vector<std::string> out;
  if (s.empty()) return out;
  for (absl::string_view p : absl::StrSplit(s, ',')) {
    if (!p.empty()) out.emplace_back(p);
  }
  return out;
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

// dune_pwo3_teacher_audit.cc MakePinnedConfig, with the fixed_800 tier shape.
DuneSearchConfig MakePinnedConfig(uint64_t search_rng_seed) {
  DuneSearchConfig c;
  c.max_simulations = absl::GetFlag(FLAGS_recovery_reference_max_simulations);
  c.relative_time_budget_ms = std::numeric_limits<double>::infinity();
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
  c.covered_prior_threshold = absl::GetFlag(FLAGS_covered_prior_threshold);
  c.conservative_stability_checkpoint_fraction =
      absl::GetFlag(FLAGS_stability_checkpoint_fraction);
  c.seed = search_rng_seed;
  c.fixed_continuation_reserve = 0;
  c.purchase_combat_budget = 16;
  c.live_continuation_reserve_seconds = 0.0;
  c.fixed_session_limit = c.max_simulations;
  return c;
}

json::Array ToJsonArray(const std::vector<Action>& v) {
  json::Array a;
  for (Action x : v) a.push_back(static_cast<int64_t>(x));
  return a;
}

json::Array ToJsonArray(const std::vector<std::string>& v) {
  json::Array a;
  for (const std::string& s : v) a.push_back(s);
  return a;
}

// The NVIDIA kernel-module version, read from procfs. nvidia-smi would be a
// subprocess whose output this tool would be transcribing; /proc/driver/nvidia/
// version is the driver reporting itself.
std::string DriverVersionLine() {
  std::ifstream f("/proc/driver/nvidia/version");
  if (!f) return "unavailable";
  std::string line;
  std::getline(f, line);
  return line;
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  at::set_num_threads(1);

  const int num_threads = absl::GetFlag(FLAGS_threads);
  if (num_threads != 1) {
    std::cerr << "STOP: --threads must be 1. The recovery instrument is pinned "
                 "single-threaded so two fresh-process runs are byte-identical "
                 "(battery check 1).\n";
    return 1;
  }
  const std::string engine_tree = absl::GetFlag(FLAGS_engine_tree_hash);
  if (engine_tree.empty()) {
    std::cerr << "STOP: --engine_tree_hash is REQUIRED. The recovered priors are "
                 "only meaningful against the frozen engine that produced the "
                 "corpus histories.\n";
    return 1;
  }
  const std::string out_path = absl::GetFlag(FLAGS_output_path);
  if (out_path.empty()) {
    std::cerr << "STOP: --output_path is required.\n";
    return 1;
  }

  const std::string self_sha = pwo2::Sha256File(argv[0]);
  auto game = LoadGame("dune_imperium");
  const int64_t obs_size = game->GetType().provides_information_state_tensor
                               ? game->InformationStateTensorSize()
                               : game->ObservationTensorSize();
  const int64_t act_size = game->NumDistinctActions();
  // Automatic device selection, identical to the audit tool's. On this box it
  // selects CUDA; the manifests never recorded that, which is why the
  // amendment's device disclosure exists.
  torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                                     : torch::Device(torch::kCPU);
  const std::string ck_a = absl::GetFlag(FLAGS_branch_a_checkpoint);
  auto model_a = LoadModel(ck_a, obs_size, act_size, device);
  const std::string sha_a = pwo2::Sha256File(ck_a);
  const float cap = static_cast<float>(absl::GetFlag(FLAGS_candidate_logit_cap));

  std::string gpu_name = "n/a";
  std::string compute_capability = "n/a";
  if (device.is_cuda()) {
    const auto* props = at::cuda::getDeviceProperties(0);
    gpu_name = props->name;
    compute_capability =
        absl::StrFormat("%d.%d", props->major, props->minor);
  }

  std::cerr << "dune_pwo3_prior_recovery threads=" << num_threads
            << " device=" << (device.is_cuda() ? "CUDA" : "CPU")
            << "\n  binary sha256 " << self_sha
            << "\n  branch_a      " << sha_a
            << "\n  gpu           " << gpu_name
            << "\n  driver        " << DriverVersionLine()
            << "\n  libtorch      " << TORCH_VERSION
            << "\n  logit cap     " << cap << "\n";

  std::vector<pwo3_recovery::Root> roots;
  const auto want = SplitCsv(absl::GetFlag(FLAGS_corpora));
  auto wants = [&](const std::string& c) {
    return std::find(want.begin(), want.end(), c) != want.end();
  };
  if (wants("main"))
    pwo3_recovery::LoadCorpusInto(absl::GetFlag(FLAGS_main_corpus), "main", &roots);
  if (wants("conversion"))
    pwo3_recovery::LoadCorpusInto(absl::GetFlag(FLAGS_conversion_corpus),
                                  "conversion", &roots);
  if (roots.empty()) {
    std::cerr << "STOP: no roots loaded.\n";
    return 1;
  }
  // Canonical emission order, independent of how the corpus files happen to be
  // concatenated. Duplicate root ids across corpora would make "one recovery per
  // root" ambiguous, so they are a hard stop rather than a silent last-wins.
  std::sort(roots.begin(), roots.end(),
            [](const pwo3_recovery::Root& a, const pwo3_recovery::Root& b) {
              return a.root_id < b.root_id;
            });
  for (size_t i = 1; i < roots.size(); ++i) {
    if (roots[i].root_id == roots[i - 1].root_id) {
      std::cerr << "STOP: root id " << roots[i].root_id
                << " appears in more than one corpus.\n";
      return 1;
    }
  }
  std::cerr << "  roots: " << roots.size() << "\n";

  std::ofstream out(out_path);
  if (!out) {
    std::cerr << "cannot open --output_path\n";
    return 1;
  }

  // ---- Record 0: the provenance block. NO timestamp: the data file must be
  // byte-identical across two fresh-process runs. -----------------------------
  {
    json::Object p;
    p["record"] = std::string("provenance");
    p["tool"] = std::string("dune_pwo3_prior_recovery");
    p["amendment"] = std::string("docs/PWO3_AMENDMENT_2_PRECISION_RECOVERY.md");
    p["recovery_binary_sha256"] = self_sha;
    p["model_sha256"] = sha_a;
    p["model_path"] = ck_a;
    p["engine_tree_hash"] = engine_tree;
    p["device"] = std::string(device.is_cuda() ? "CUDA" : "CPU");
    p["gpu_name"] = gpu_name;
    p["gpu_compute_capability"] = compute_capability;
    p["driver_version"] = DriverVersionLine();
    p["libtorch_version"] = std::string(TORCH_VERSION);
    p["cuda_runtime_version"] =
        static_cast<int64_t>(at::detail::getCUDAHooks().versionCUDART());
    p["cudnn_version"] =
        static_cast<int64_t>(at::detail::getCUDAHooks().versionCuDNN());
    p["torch_cuda_config"] = at::detail::getCUDAHooks().showConfig();
    // AutocastGuard(device_.type(), device_.is_cuda()) in PriorAndEvaluate:
    // autocast is ENABLED exactly when the evaluator runs on CUDA.
    p["autocast_enabled"] = device.is_cuda();
    p["autocast_note"] = std::string(
        "DuneNNEvaluator::PriorAndEvaluate wraps the forward pass in "
        "AutocastGuard(device.type(), device.is_cuda()): FP16 autocast on CUDA, "
        "then an explicit FP32 cast before a CPU DOUBLE softmax "
        "(dune_evaluator.h:156-234).");
    p["threads"] = static_cast<int64_t>(num_threads);
    p["at_num_threads"] = static_cast<int64_t>(at::get_num_threads());
    p["candidate_logit_cap"] = static_cast<double>(cap);
    p["simulations_per_root"] = static_cast<int64_t>(0);
    p["reference_tier_max_simulations"] =
        static_cast<int64_t>(absl::GetFlag(FLAGS_recovery_reference_max_simulations));
    p["reference_search_seed"] =
        static_cast<int64_t>(absl::GetFlag(FLAGS_recovery_reference_seed));
    p["n_roots"] = static_cast<int64_t>(roots.size());
    p["emission_order"] = std::string("root_id ascending");
    p["evaluator_entry_point"] = std::string(
        "DuneNNEvaluator::PriorAndEvaluate (NOT Prior: a single-row batch is "
        "different numerics and feeds the `priors` branch, which zero PWO-3 "
        "rows took)");
    p["chain"] = std::string(
        "LoadCorpusInto -> Reconstruct -> RunSearch(state, 0 sims, inf) -> "
        "InitializePriorsAndValue -> PriorAndEvaluate -> child.raw_prior -> "
        "GetRootDiagnostics -> FilterAndNormalizeRawPriors -> diag.raw_priors");
    out << json::ToString(p) << "\n";
  }

  int n_raw_priors_branch = 0;
  int n_priors_branch = 0;
  const auto t0 = std::chrono::steady_clock::now();

  auto ev_a = std::make_shared<DuneNNEvaluator>(model_a, device, cap);
  std::shared_ptr<algorithms::Evaluator> ev =
      std::static_pointer_cast<algorithms::Evaluator>(ev_a);

  for (size_t i = 0; i < roots.size(); ++i) {
    const pwo3_recovery::Root& r = roots[i];
    auto state = pwo3_recovery::Reconstruct(game, r.history);
    SPIEL_CHECK_EQ(state->CurrentPlayer(), r.player);

    const uint64_t rng_seed = pwo2::SearchRngSeed(
        absl::GetFlag(FLAGS_recovery_reference_seed), r.root_id);
    DuneSearchConfig cfg = MakePinnedConfig(rng_seed);
    DunePUCTISMCTSBot bot(cfg, ev);
    // ZERO simulations. RunSearch still creates the root node and calls
    // InitializePriorsAndValue -- the expansion whose raw_prior writes are the
    // whole point -- and then runs a loop of length 0.
    DuneSearchResult res = bot.RunSearch(
        *state, /*max_sims=*/0,
        /*max_time_ms=*/std::numeric_limits<double>::infinity(),
        /*start_sim_index=*/0);
    SPIEL_CHECK_EQ(res.simulations_completed, 0);

    SearchDiagnostics d =
        bot.GetRootDiagnostics(*state, cfg.min_visit_threshold, kInvalidAction);
    SPIEL_CHECK_EQ(d.actions.size(), r.legal_actions.size());
    SPIEL_CHECK_TRUE(pwo3::IsAscending(d.actions));
    SPIEL_CHECK_EQ(d.total_root_visits, 0);

    // Which branch of GetRootDiagnostics produced the vector. Every PWO-3 grid
    // row carries raw_prior_source == "raw_priors"; the battery treats anything
    // else as a FAIL even if the numbers would project correctly.
    const bool from_raw_priors = (d.raw_priors.size() == d.actions.size());
    const std::vector<double>& recovered =
        from_raw_priors ? d.raw_priors : d.priors;
    SPIEL_CHECK_EQ(recovered.size(), d.actions.size());
    if (from_raw_priors) {
      ++n_raw_priors_branch;
    } else {
      ++n_priors_branch;
    }

    std::vector<std::string> dec, hex, legacy;
    dec.reserve(recovered.size());
    hex.reserve(recovered.size());
    legacy.reserve(recovered.size());
    double sum = 0.0;
    for (double v : recovered) {
      dec.push_back(FmtExact(v));
      hex.push_back(FmtHex(v));
      legacy.push_back(LegacyProjection(v));
      sum += v;
    }

    json::Object o;
    o["record"] = std::string("root");
    o["root_id"] = r.root_id;
    o["corpus"] = r.corpus;
    o["stratum"] = r.stratum;
    o["half"] = r.half;
    o["source_arm"] = r.source_arm;
    o["source_episode_id"] = static_cast<int64_t>(r.source_episode_id);
    o["acting_player"] = static_cast<int64_t>(r.player);
    o["n_legal"] = static_cast<int64_t>(d.actions.size());
    o["root_actions"] = ToJsonArray(d.actions);
    o["branch"] = std::string(from_raw_priors ? "raw_priors" : "priors");
    // THREE representations of the same vector. `recovered_prior_decimal` is
    // the one the analyzer consumes; the hex form is the parser-independent
    // check; `legacy_projection` is what json.cc WOULD have written, and is what
    // battery check 4 compares against the stored value.
    o["recovered_prior_decimal"] = ToJsonArray(dec);
    o["recovered_prior_hex"] = ToJsonArray(hex);
    o["legacy_projection"] = ToJsonArray(legacy);
    o["recovered_prior_sum_decimal"] = FmtExact(sum);
    o["recovered_argmax_action"] =
        static_cast<int64_t>(pwo3::ArgmaxLowestId(d.actions, recovered));
    o["simulations_completed"] = static_cast<int64_t>(res.simulations_completed);
    o["total_root_visits"] = static_cast<int64_t>(d.total_root_visits);
    out << json::ToString(o) << "\n";

    if ((i + 1) % 25 == 0 || i + 1 == roots.size())
      std::cerr << "  " << (i + 1) << "/" << roots.size() << "\r" << std::flush;
  }

  const double elapsed_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::cerr << "\n  recovery done: " << roots.size() << " roots, "
            << n_raw_priors_branch << " via raw_priors, " << n_priors_branch
            << " via priors, " << absl::StrFormat("%.1f", elapsed_s) << " s\n";
  if (n_priors_branch != 0) {
    std::cerr << "WARNING: " << n_priors_branch
              << " root(s) took the `priors` branch. Every PWO-3 grid row was "
                 "written from `raw_priors`; the recovery gate treats this as a "
                 "battery FAILURE (check 6).\n";
  }
  return 0;
}
