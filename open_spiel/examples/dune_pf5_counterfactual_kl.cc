// PF-5 item 3 / PF-6 section 6 — counterfactual held-out teacher KL.
//
// docs/PWO5_POSTMORTEM_ADDENDUM.md section 3.
//
// ---------------------------------------------------------------------------
// WHAT IT ANSWERS
// ---------------------------------------------------------------------------
//
// The addendum's section 1 shows held-out teacher KL ending HIGHER than it
// started in every TREATED arm (T +0.09/+0.14, H +0.47/+0.50). That is a
// description, not a diagnosis, because nobody measured what KL did in an arm
// carrying NO distillation term at all. If the untreated P arms diverge just
// as much, divergence is what training does and the treatment explains
// nothing; if they do not, the treatment moved something.
//
// This binary computes that counterfactual on the P1/P2 checkpoints.
//
// ---------------------------------------------------------------------------
// WHY IT REUSES ComputeValidationKL RATHER THAN RE-DERIVING THE STATISTIC
// ---------------------------------------------------------------------------
//
// The number in `diagnostics.csv`'s `validation_kl` column -- the number the
// addendum's section 1 table is built from -- is produced by exactly one
// function: SearchLabelBuffer::ComputeValidationKL
// (dune_search_label_buffer.h:363). A re-implementation would have to
// reproduce logit centring and capping, the -1e9 mask fill, the 1e-12 teacher
// clamp, the sum-over-actions-then-mean-over-rows reduction, and the fp32
// accumulation -- and any drift in any of them would show up as a
// treated-vs-untreated DIFFERENCE, which is precisely the quantity under
// test. So this binary calls the same function on the same buffer.
//
// The comparison is therefore population-identical AND statistic-identical:
//
//   * POPULATION. The pack at calibration_results_v2/pwo5_prep/search_labels
//     is role-aware; `role=validation` is, verbatim from its manifest, "the
//     weight-one rows of the 60 PWO-5 section 8.5 HELD-OUT games". That is
//     game-disjoint. The legacy %11 per-row split is NOT used and is not
//     reachable from here -- LoadFromDirectory honours declared roles, and a
//     role-aware directory refuses the rescan path.
//   * STATISTIC. Same function, same logit_cap (all six PWO-5 arms ran
//     --logit_cap=10.0; it is also the flag default).
//
// ---------------------------------------------------------------------------
// WHY IT IS NOT A TRAINER INVOCATION
// ---------------------------------------------------------------------------
//
// Even at --learning_rate=0 the trainer collects an on-policy rollout before
// optimizing (dune_ppo_train.cc:3169 "Collect first rollout synchronously",
// executing at :3173-3175, with the update loop only opening at :3208). That
// would GENERATE GAMES, violating PF-5's no-new-games constraint outright.
// This binary loads checkpoints, runs forward passes under NoGradGuard, and
// writes JSON. It constructs no optimizer, no rollout, and no game.
//
// It selects CUDA when available, so it runs under section 0.6 exclusivity.
//
// Exit: 0 computed, 1 a registered assertion failed, 2 operational.

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/spiel.h"
#include "dune_network.h"
#include "dune_search_label_buffer.h"

ABSL_FLAG(std::string, game, "dune_imperium", "Must match the checkpoint.");
ABSL_FLAG(std::string, model_checkpoint, "", "REQUIRED. Read-only.");
ABSL_FLAG(std::string, search_label_dir,
          "calibration_results_v2/pwo5_prep/search_labels",
          "Role-aware pack. Its `validation` role IS the section 8.5 "
          "game-disjoint held-out population.");
ABSL_FLAG(int, hidden_dim, 2048, "Must match the checkpoint.");
ABSL_FLAG(int, num_blocks, 8, "Must match the checkpoint.");
ABSL_FLAG(bool, nonlinear_value_head, false, "Must match the checkpoint.");
ABSL_FLAG(bool, with_aux_heads, true,
          "PWO-5 pilot checkpoints (P/T/H alike) carry the three auxiliary "
          "heads. torch::load fails loudly if this is wrong.");
ABSL_FLAG(double, logit_cap, 10.0,
          "MUST equal the value the arm trained under, or the statistic is "
          "not the one diagnostics.csv reports. All six PWO-5 arms ran 10.0.");
ABSL_FLAG(int, expect_validation_rows, 3541,
          "Registered. The pack manifest records validation_label_count=3541. "
          "Asserted, not assumed -- a silently smaller population would move "
          "the KL without moving anything visible.");
ABSL_FLAG(std::string, output_json, "", "REQUIRED.");
ABSL_FLAG(std::string, arm_label, "", "Recorded verbatim.");
ABSL_FLAG(int, update, -1, "Recorded verbatim (the checkpoint's update).");

// ---------------------------------------------------------------------------
// LINK-SATISFACTION ONLY. NOT KNOBS OF THIS MEASUREMENT.
// ---------------------------------------------------------------------------
//
// This binary links dune_ppo_training_utils.cc to get CenterAndCapLogitsTensor
// -- the transform SearchLabelBuffer::ComputeValidationKL calls -- rather than
// copying that function, since a copy is precisely the re-derivation this tool
// exists to avoid. That object also contains TrainPpoUpdate, which references
// these flags, so the linker demands definitions for them.
//
// NONE of them is read on any path this binary executes: TrainPpoUpdate is
// never called here, no optimizer is constructed, and no gradient is taken.
// The only flag that reaches the computation is --logit_cap above, which is
// passed explicitly to ComputeValidationKL. Values below are the trainer's
// defaults and are inert.
//
// dune_ppo_training_utils_test.cc carries the same block for the same reason.
ABSL_FLAG(int, ppo_minibatch_size, 2048, "INERT here (link-satisfaction).");
ABSL_FLAG(int, ppo_update_epochs, 4, "INERT here (link-satisfaction).");
ABSL_FLAG(double, ppo_clip_epsilon, 0.2, "INERT here (link-satisfaction).");
ABSL_FLAG(bool, normalize_advantages, true, "INERT here (link-satisfaction).");
ABSL_FLAG(bool, ppo_clip_value_loss, true, "INERT here (link-satisfaction).");
ABSL_FLAG(double, entropy_coef, 0.01, "INERT here (link-satisfaction).");
ABSL_FLAG(double, value_coef, 0.5, "INERT here (link-satisfaction).");
ABSL_FLAG(double, target_kl, 0.0, "INERT here (link-satisfaction).");
ABSL_FLAG(bool, train_amp, true, "INERT here (link-satisfaction).");
ABSL_FLAG(double, grad_clip_norm, 0.5, "INERT here (link-satisfaction).");
ABSL_FLAG(uint64_t, shaping_start_env_steps, 206830543,
          "INERT here (link-satisfaction).");
ABSL_FLAG(uint64_t, shaping_decay_env_steps, 0,
          "INERT here (link-satisfaction).");
ABSL_FLAG(bool, diagnostics_only, false, "INERT here (link-satisfaction).");

namespace {

std::string JsonEscape(const std::string& s) {
  std::string out;
  for (char c : s) {
    if (c == '"' || c == '\\') { out.push_back('\\'); out.push_back(c); }
    else out.push_back(c);
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  const std::string ckpt = absl::GetFlag(FLAGS_model_checkpoint);
  const std::string out_path = absl::GetFlag(FLAGS_output_json);
  if (ckpt.empty() || out_path.empty()) {
    std::cerr << "--model_checkpoint and --output_json are REQUIRED.\n";
    return 2;
  }
  {
    std::ifstream probe(ckpt, std::ios::binary);
    if (!probe) {
      std::cerr << "checkpoint not readable: " << ckpt << "\n";
      return 2;
    }
  }

  auto game = open_spiel::LoadGame(absl::GetFlag(FLAGS_game));
  const int64_t obs_size = game->GetType().provides_information_state_tensor
                               ? game->InformationStateTensorSize()
                               : game->ObservationTensorSize();
  const int64_t action_size = game->NumDistinctActions();

  open_spiel::SearchLabelBuffer buffer;
  buffer.SetExpectedDimensions(obs_size, action_size);
  buffer.LoadFromDirectory(absl::GetFlag(FLAGS_search_label_dir));

  const int64_t val_rows = static_cast<int64_t>(buffer.ValidationSize());
  const int64_t expect = absl::GetFlag(FLAGS_expect_validation_rows);
  if (!buffer.role_aware()) {
    std::cerr << "REGISTERED ASSERTION FAILED: the pack at "
              << absl::GetFlag(FLAGS_search_label_dir)
              << " is NOT role-aware, so its validation split would be the "
                 "legacy %11 per-row hash. Those rows share games with the "
                 "training rows, and calling them held out is wrong at the "
                 "level that matters.\n";
    return 1;
  }
  if (val_rows != expect) {
    std::cerr << "REGISTERED ASSERTION FAILED: validation rows " << val_rows
              << " != expected " << expect << ". The population moved.\n";
    return 1;
  }

  const torch::Device device(torch::cuda::is_available() ? torch::kCUDA
                                                         : torch::kCPU);
  auto model = std::make_shared<open_spiel::SharedDunePolicyValueNetImpl>(
      obs_size, absl::GetFlag(FLAGS_hidden_dim), action_size,
      absl::GetFlag(FLAGS_num_blocks),
      absl::GetFlag(FLAGS_nonlinear_value_head),
      absl::GetFlag(FLAGS_with_aux_heads), /*head_init_seed=*/0);
  torch::load(model, ckpt, device);
  model->to(device);
  // ResBlocks use LayerNorm, so this is numerically a no-op and the statistic
  // matches the trainer's in-training call exactly. Kept as hygiene against a
  // future Dropout, matching dune_pwo5_head_eval.cc:602.
  model->eval();

  const float cap = static_cast<float>(absl::GetFlag(FLAGS_logit_cap));
  const double kl = buffer.ComputeValidationKL(model, device, cap);

  std::ostringstream js;
  js << "{\n"
     << "  \"arm\": \"" << JsonEscape(absl::GetFlag(FLAGS_arm_label)) << "\",\n"
     << "  \"update\": " << absl::GetFlag(FLAGS_update) << ",\n"
     << "  \"validation_kl\": " << kl << ",\n"
     << "  \"validation_rows\": " << val_rows << ",\n"
     << "  \"logit_cap\": " << absl::GetFlag(FLAGS_logit_cap) << ",\n"
     << "  \"role_aware_pack\": true,\n"
     << "  \"population\": \"PWO-5 section 8.5 held-out games "
        "(game-disjoint); pack role=validation\",\n"
     << "  \"statistic\": \"SearchLabelBuffer::ComputeValidationKL -- the same "
        "function that writes diagnostics.csv validation_kl\",\n"
     << "  \"model_checkpoint\": \"" << JsonEscape(ckpt) << "\",\n"
     << "  \"search_label_dir\": \""
     << JsonEscape(absl::GetFlag(FLAGS_search_label_dir)) << "\",\n"
     << "  \"device\": \"" << (device.is_cuda() ? "cuda" : "cpu") << "\"\n"
     << "}\n";

  std::ofstream ofs(out_path);
  if (!ofs) {
    std::cerr << "cannot write " << out_path << "\n";
    return 2;
  }
  ofs << js.str();
  ofs.close();

  std::cout << absl::GetFlag(FLAGS_arm_label) << " u"
            << absl::GetFlag(FLAGS_update) << "  validation_kl=" << kl
            << "  (rows=" << val_rows << ", cap=" << cap << ")\n";
  return 0;
}
