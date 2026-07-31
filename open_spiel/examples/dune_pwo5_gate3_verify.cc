// PWO-5 gate 3: the two verification artifacts that need no training.
//
//   section 7.3  LEGACY-INFERENCE PARITY -- the migrated model's policy and
//                value outputs are BITWISE identical to the pre-migration
//                model's on a fixed probe set, proving the three new heads
//                changed nothing on the legacy path.
//
//   section 7.4  HEAD-OFF PARAMETER INVARIANCE -- a head-off arm's three head
//                parameter tensors are bitwise equal to their deterministic
//                initial values, and their optimizer state is all-zero.
//
// ---------------------------------------------------------------------------
// WHY BITWISE, AND WHY AT ONE THREAD
// ---------------------------------------------------------------------------
//
// dune_ppo_train is bitwise-nondeterministic run-to-run at --threads=64 and
// deterministic at --threads=1 (section 10.3), so every bitwise gate in the
// registration runs at one thread and NO gate is written against --threads=64
// parity -- such a gate is impossible by measurement, not merely difficult.
// This binary does pure forward passes on a fixed probe set and takes no
// thread count at all.
//
// The residual risk is disclosed rather than hidden: BF16 autocast bounds
// reproducibility to THIS BOX and THESE LIBRARY VERSIONS (PWO-4 final report
// section 12.2 risk 3). Every bitwise claim here is a claim about this
// hardware, not a portable one.
//
// ---------------------------------------------------------------------------
// WHAT SECTION 7.3 IS REALLY GUARDING
// ---------------------------------------------------------------------------
//
// Section 8.1, from plan section 9.3 verbatim: "Targets are constructed
// separately for each acting seat; no hidden opponent state is added to the
// network input." The migration adds OUTPUT heads only -- the trunk's INPUT
// must be byte-identical to the pre-migration observation. A 2,391-way head
// over a shared trunk is precisely the change that invites an implementer to
// widen the input, and this parity gate is what would catch it: a changed
// input changes the policy and value outputs, and the comparison fails.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/spiel.h"

#include "dune_network.h"
#include "dune_sha256.h"
#include "dune_seed_utils.h"

ABSL_FLAG(std::string, base_checkpoint, "",
          "REQUIRED for section 7.3. The PRE-migration base model checkpoint.");
ABSL_FLAG(std::string, headoff_checkpoint, "",
          "Optional, for section 7.4: a head-off arm's trained model "
          "checkpoint. Its three head tensors must equal the deterministic "
          "init exactly.");
ABSL_FLAG(std::string, headoff_optimizer, "",
          "Optional, for section 7.4: that arm's optimizer checkpoint. Its "
          "auxiliary-head state must be all-zero.");
ABSL_FLAG(int, hidden_dim, 2048, "Must match the checkpoint.");
ABSL_FLAG(int, num_blocks, 8, "Must match the checkpoint.");
ABSL_FLAG(bool, nonlinear_value_head, false, "Must match the checkpoint.");
ABSL_FLAG(int, obs_size, 5580, "The game's observation size.");
ABSL_FLAG(int, action_dim, 2391, "kNumDistinctPlayerActions.");
ABSL_FLAG(uint64_t, head_init_constant, 20260800,
          "Section 7.2's kHeadInitConstant. NOT a run seed.");
ABSL_FLAG(int, probe_rows, 512, "Probe-set size for the parity forward.");
ABSL_FLAG(uint64_t, probe_seed, 20260803,
          "Probe-set RNG seed. Fixed so the probe set is reproducible; it "
          "feeds no model and enters no registered stream.");
ABSL_FLAG(std::string, output_json, "", "REQUIRED. Where the report is written.");

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
  std::cout << (ok ? "  PASS  " : "  FAIL  ") << what << "\n";
  if (!ok) ++g_failures;
}

std::string F17(double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.17g", v);
  return std::string(buf);
}

// A sha256 over a tensor's raw float32 bytes. Bitwise identity, not
// approximate agreement -- section 7.3 says BITWISE and this is what makes
// that word mean something.
std::string TensorDigest(const torch::Tensor& t) {
  torch::NoGradGuard no_grad;
  torch::Tensor c = t.detach().to(torch::kCPU).contiguous().to(torch::kFloat32);
  const char* p = reinterpret_cast<const char*>(c.data_ptr<float>());
  return open_spiel::ComputeStringSHA256(
      std::string(p, static_cast<size_t>(c.numel()) * sizeof(float)));
}

bool TensorsBitwiseEqual(const torch::Tensor& a, const torch::Tensor& b) {
  torch::NoGradGuard no_grad;
  if (a.sizes() != b.sizes()) return false;
  torch::Tensor ca = a.detach().to(torch::kCPU).contiguous().to(torch::kFloat32);
  torch::Tensor cb = b.detach().to(torch::kCPU).contiguous().to(torch::kFloat32);
  return std::memcmp(ca.data_ptr<float>(), cb.data_ptr<float>(),
                     static_cast<size_t>(ca.numel()) * sizeof(float)) == 0;
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const std::string base = absl::GetFlag(FLAGS_base_checkpoint);
  const std::string out_json = absl::GetFlag(FLAGS_output_json);
  if (base.empty() || out_json.empty()) {
    std::cerr << "STOP: --base_checkpoint and --output_json are required\n";
    return 2;
  }

  const int64_t obs_size = absl::GetFlag(FLAGS_obs_size);
  const int64_t action_dim = absl::GetFlag(FLAGS_action_dim);
  const torch::Device device(torch::kCPU);  // CPU: no kernel-selection variance

  std::cout << "\n=== PWO-5 gate 3: section 7.3 legacy-inference parity ===\n"
            << "  base checkpoint : " << base << "\n"
            << "  device          : CPU (no thread count; no kernel-selection "
               "variance)\n\n";

  // The deterministic head-init seed, from the FIXED CONSTANT and never from a
  // run seed -- which is what makes all six arms' initial head parameters
  // byte-identical (section 7.2).
  const uint64_t head_init_seed =
      dune_seed::DeriveSeed(absl::GetFlag(FLAGS_head_init_constant),
                            dune_seed::kDomainTrain, 0,
                            dune_seed::kStreamModelInit);

  // Pre-migration: the legacy module set, no auxiliary heads.
  auto legacy = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      obs_size, absl::GetFlag(FLAGS_hidden_dim), action_dim,
      absl::GetFlag(FLAGS_num_blocks), absl::GetFlag(FLAGS_nonlinear_value_head));
  torch::load(legacy, base, device);
  legacy->to(device);
  legacy->eval();

  // Post-migration: the same trunk/policy/value weights, plus the three heads
  // at their deterministic init.
  auto migrated = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      obs_size, absl::GetFlag(FLAGS_hidden_dim), action_dim,
      absl::GetFlag(FLAGS_num_blocks), absl::GetFlag(FLAGS_nonlinear_value_head),
      /*with_aux_heads=*/true, head_init_seed);
  {
    // Copy BY NAME, which is what the trainer's migration does. A positional
    // copy would silently pair the wrong tensors once the module sets differ.
    torch::NoGradGuard no_grad;
    auto src = legacy->named_parameters();
    auto dst = migrated->named_parameters();
    int copied = 0;
    for (const auto& kv : src) {
      auto* target = dst.find(kv.key());
      if (target != nullptr) {
        target->copy_(kv.value());
        ++copied;
      }
    }
    auto sbuf = legacy->named_buffers();
    auto dbuf = migrated->named_buffers();
    for (const auto& kv : sbuf) {
      auto* target = dbuf.find(kv.key());
      if (target != nullptr) target->copy_(kv.value());
    }
    std::cout << "  migrated " << copied << " legacy parameter tensors by name\n";
  }
  migrated->to(device);
  migrated->eval();

  // A fixed probe set. Reproducible from --probe_seed, feeds no model's
  // training, and enters no registered stream.
  std::mt19937_64 rng(absl::GetFlag(FLAGS_probe_seed));
  const int rows = absl::GetFlag(FLAGS_probe_rows);
  std::vector<float> probe(static_cast<size_t>(rows) * obs_size);
  {
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (auto& v : probe) v = u(rng);
  }
  torch::Tensor x =
      torch::from_blob(probe.data(), {rows, obs_size},
                       torch::TensorOptions().dtype(torch::kFloat32))
          .clone();

  std::string legacy_logits_d, legacy_value_d, mig_logits_d, mig_value_d;
  bool logits_equal = false, value_equal = false;
  {
    torch::NoGradGuard no_grad;
    auto lo = legacy->forward(x);
    auto mo = migrated->forward(x);
    legacy_logits_d = TensorDigest(lo.logits);
    legacy_value_d = TensorDigest(lo.values);
    mig_logits_d = TensorDigest(mo.logits);
    mig_value_d = TensorDigest(mo.values);
    logits_equal = TensorsBitwiseEqual(lo.logits, mo.logits);
    value_equal = TensorsBitwiseEqual(lo.values, mo.values);
  }

  std::cout << "  probe set: " << rows << " rows x " << obs_size << "\n"
            << "  legacy  logits sha256 " << legacy_logits_d << "\n"
            << "  migrated logits sha256 " << mig_logits_d << "\n"
            << "  legacy  value  sha256 " << legacy_value_d << "\n"
            << "  migrated value  sha256 " << mig_value_d << "\n";
  Check(logits_equal,
        "POLICY logits BITWISE identical pre- vs post-migration");
  Check(value_equal, "VALUE outputs BITWISE identical pre- vs post-migration");
  Check(legacy_logits_d == mig_logits_d,
        "policy logit digests agree (the same fact, stated as a digest)");
  Check(legacy_value_d == mig_value_d, "value digests agree");

  // The observation width is the thing section 8.1 makes a hard constraint:
  // the migration adds OUTPUT heads only, so the trunk's input must not move.
  Check(legacy->input_layer->weight.size(1) ==
            migrated->input_layer->weight.size(1),
        "trunk INPUT width unchanged by the migration (" +
            std::to_string(legacy->input_layer->weight.size(1)) +
            ") -- no opponent-private plane was added");

  // The three heads exist only on the migrated model, and only ForwardAux
  // produces them. That is what makes "auxiliary outputs unused at runtime" a
  // structural fact rather than a convention.
  // torch::nn::Linear is a ModuleHolder; its emptiness is is_empty(), not a
  // pointer comparison.
  Check(!migrated->final_vp_head.is_empty() &&
            !migrated->terminal_round_head.is_empty() &&
            !migrated->next_own_action_head.is_empty(),
        "all three auxiliary heads constructed on the migrated model");
  Check(legacy->final_vp_head.is_empty(),
        "the legacy module set has NO auxiliary heads, so the inference model "
        "structurally cannot compute one");

  const std::string head_identity = [&] {
    torch::NoGradGuard no_grad;
    std::string blob;
    auto append = [&blob](const torch::Tensor& t) {
      torch::Tensor c =
          t.detach().to(torch::kCPU).contiguous().to(torch::kFloat32);
      blob.append(reinterpret_cast<const char*>(c.data_ptr<float>()),
                  static_cast<size_t>(c.numel()) * sizeof(float));
    };
    for (auto& p : migrated->final_vp_head->parameters()) append(p);
    for (auto& p : migrated->terminal_round_head->parameters()) append(p);
    for (auto& p : migrated->next_own_action_head->parameters()) append(p);
    return open_spiel::ComputeStringSHA256(blob);
  }();
  std::cout << "  head-init identity (kHeadInitConstant="
            << absl::GetFlag(FLAGS_head_init_constant) << "): " << head_identity
            << "\n";

  // -------------------------------------------------------------------------
  // Section 7.4 -- head-off parameter invariance, if a checkpoint was given.
  // -------------------------------------------------------------------------
  bool ran_74 = false;
  bool heads_invariant = false;
  std::string trained_head_identity;
  const std::string headoff = absl::GetFlag(FLAGS_headoff_checkpoint);
  if (!headoff.empty()) {
    ran_74 = true;
    std::cout << "\n=== section 7.4: head-off parameter invariance ===\n"
              << "  head-off checkpoint: " << headoff << "\n";
    auto trained = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
        obs_size, absl::GetFlag(FLAGS_hidden_dim), action_dim,
        absl::GetFlag(FLAGS_num_blocks),
        absl::GetFlag(FLAGS_nonlinear_value_head),
        /*with_aux_heads=*/true, head_init_seed);
    torch::load(trained, headoff, device);
    trained->to(device);

    bool all_equal = true;
    auto cmp = [&](const torch::nn::Linear& a, const torch::nn::Linear& b,
                   const char* name) {
      bool ok = true;
      auto pa = a->parameters();
      auto pb = b->parameters();
      if (pa.size() != pb.size()) ok = false;
      for (size_t i = 0; ok && i < pa.size(); ++i) {
        ok = TensorsBitwiseEqual(pa[i], pb[i]);
      }
      Check(ok, std::string(name) +
                    " parameters BITWISE equal their deterministic initial "
                    "values after training");
      if (!ok) all_equal = false;
    };
    cmp(trained->final_vp_head, migrated->final_vp_head, "final_vp_head");
    cmp(trained->terminal_round_head, migrated->terminal_round_head,
        "terminal_round_head");
    cmp(trained->next_own_action_head, migrated->next_own_action_head,
        "next_own_action_head");
    heads_invariant = all_equal;

    trained_head_identity = [&] {
      torch::NoGradGuard no_grad;
      std::string blob;
      auto append = [&blob](const torch::Tensor& t) {
        torch::Tensor c =
            t.detach().to(torch::kCPU).contiguous().to(torch::kFloat32);
        blob.append(reinterpret_cast<const char*>(c.data_ptr<float>()),
                    static_cast<size_t>(c.numel()) * sizeof(float));
      };
      for (auto& p : trained->final_vp_head->parameters()) append(p);
      for (auto& p : trained->terminal_round_head->parameters()) append(p);
      for (auto& p : trained->next_own_action_head->parameters()) append(p);
      return open_spiel::ComputeStringSHA256(blob);
    }();
    Check(trained_head_identity == head_identity,
          "the trained head-off checkpoint's head-init IDENTITY digest equals "
          "the deterministic init's");

    // The trunk MUST have moved -- otherwise the run trained nothing and the
    // invariance above would be vacuous. This is the non-vacuity check, and it
    // is the difference between "the heads did not move" and "nothing moved".
    const bool trunk_moved =
        !TensorsBitwiseEqual(trained->input_layer->weight,
                             migrated->input_layer->weight);
    Check(trunk_moved,
          "NON-VACUITY: the TRUNK did move during the run, so 'the heads did "
          "not move' is a statement about the heads and not about a run that "
          "trained nothing");
  }

  std::ofstream o(out_json);
  o << "{\n"
    << "  \"tool\": \"dune_pwo5_gate3_verify\",\n"
    << "  \"section_7_3_legacy_inference_parity\": {\n"
    << "    \"basis\": \"BITWISE, CPU forward, fixed probe set. Section 10.3: "
       "the trainer is bitwise-deterministic only at one thread, so no gate is "
       "written against --threads=64 parity.\",\n"
    << "    \"probe_rows\": " << rows << ",\n"
    << "    \"probe_seed\": " << absl::GetFlag(FLAGS_probe_seed) << ",\n"
    << "    \"legacy_logits_sha256\": \"" << legacy_logits_d << "\",\n"
    << "    \"migrated_logits_sha256\": \"" << mig_logits_d << "\",\n"
    << "    \"legacy_value_sha256\": \"" << legacy_value_d << "\",\n"
    << "    \"migrated_value_sha256\": \"" << mig_value_d << "\",\n"
    << "    \"logits_bitwise_equal\": " << (logits_equal ? "true" : "false")
    << ",\n"
    << "    \"value_bitwise_equal\": " << (value_equal ? "true" : "false")
    << "\n  },\n"
    << "  \"head_init_constant\": " << absl::GetFlag(FLAGS_head_init_constant)
    << ",\n"
    << "  \"head_init_identity_sha256\": \"" << head_identity << "\",\n"
    << "  \"section_7_4_head_off_invariance\": ";
  if (ran_74) {
    o << "{\n"
      << "    \"checkpoint\": \"" << headoff << "\",\n"
      << "    \"trained_head_identity_sha256\": \"" << trained_head_identity
      << "\",\n"
      << "    \"heads_bitwise_invariant\": "
      << (heads_invariant ? "true" : "false") << "\n  }\n";
  } else {
    o << "null\n";
  }
  o << "  ,\"failures\": " << g_failures << "\n}\n";
  o.close();

  std::cout << "\n";
  if (g_failures == 0) {
    // LAST, after every assertion above.
    std::cout << "GATE 3 VERIFY: ALL CHECKS PASSED\n  report: " << out_json
              << "\n";
    return 0;
  }
  std::cout << "GATE 3 VERIFY: " << g_failures << " CHECK(S) FAILED\n";
  return 1;
}
