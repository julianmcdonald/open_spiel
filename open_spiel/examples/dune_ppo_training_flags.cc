// Trainer flags that dune_ppo_training_utils.cc reads.
//
// Every binary that links dune_ppo_training_utils.cc needs these defined.
// They live here, with the trainer's defaults, so dune_ppo_train and the
// tools that reuse its training utilities share one definition. The few
// tests that need different defaults define all twelve themselves and do
// not link this file. Declarations: dune_ppo_training_flags.h.

#include <cstdint>
#include <string>

#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "dune_ppo_training_flags.h"

// Rollout batch size.
ABSL_FLAG(int, rollout_games, 0,
          "Exact complete games collected per PPO update rollout batch. "
          "If > 0, overrides transition-threshold mode.");

// Network architecture and input layout.
ABSL_FLAG(int, hidden_dim, 2048, "Network hidden dimension.");
ABSL_FLAG(int, num_blocks, 8, "Network residual block count.");
ABSL_FLAG(bool, nonlinear_value_head, false, "Use a nonlinear value head.");
ABSL_FLAG(std::string, market_appendix_mode, "none",
          "Card-slot input mode: 'none' (5580), 'zeros' or 'ordered_market' (6215), "
          "or 'ordered_card_slots_v2' (7271: Imperium, Tleilaxu, and own Helena reserve).");

// Runtime numerics and action scoring.
ABSL_FLAG(bool, allow_tf32, true,
          "Allow TF32 for CUDA CuBLAS/CuDNN. Default true preserves the "
          "historical runtime policy.");
ABSL_FLAG(bool, enable_semantic_scorer, false, "Enable the neural Semantic Action Scorer on legal candidate actions.");
ABSL_FLAG(bool, enable_zone_encoder, false, "Enable the zone encoder adapter feeding trunk.");
ABSL_FLAG(std::string, semantic_descriptor_schema, "semantic_action_v3",
          "Semantic descriptor schema version (e.g. semantic_action_v3, semantic_action_v4).");

// Training-only reward shaping.
// Sign convention: the value is SUBTRACTED from reward — pass a POSITIVE value to penalize. (The phase-18 pilots passed -0.02, which was a +0.02 bonus.)
// A NEGATIVE value is now rejected fatally in main() (PWO-5 gate 2 item (a)).
ABSL_FLAG(double, specimen_exchange_penalty, 0.0,
          "Magnitude of the negative shaping SUBTRACTED from a transition that "
          "takes a ConvertSpecimenToTroop action (IDs 741-752; 740 is an unused "
          "base constant and is never legal). MUST BE >= 0: the value is "
          "subtracted, so a POSITIVE value penalizes and a NEGATIVE value would "
          "be a BONUS on the very behaviour this term exists to suppress -- a "
          "negative value is rejected fatally at startup. Training-only "
          "anti-breadcrumb (never eval). Apply the SAME value to BOTH pilot and "
          "control arms so the search-distillation contrast stays a pure "
          "experiment. Typical 0.02 (terminal win utility is 2.25). Requires "
          "--allow_shaping.");
ABSL_FLAG(double, family_atomics_penalty, 0.0,
          "Magnitude of the negative shaping SUBTRACTED from a transition that "
          "uses Family Atomics (ID 1591) when NOT in that player's own reveal turn "
          "with remaining persuasion > 0. MUST BE >= 0. Typical 0.10. Requires "
          "--allow_shaping.");
ABSL_FLAG(double, plot_intrigue_penalty, 0.0,
          "Magnitude of the negative shaping SUBTRACTED from a transition that "
          "plays a plot intrigue card (IDs 1600-1699) when the player holds fewer "
          "than plot_intrigue_exemption_threshold intrigue cards before playing. "
          "MUST BE >= 0. Typical 0.01. Requires --allow_shaping.");
ABSL_FLAG(int, plot_intrigue_exemption_threshold, 3,
          "Intrigue hand size threshold at or above which plot intrigue plays are "
          "exempt from penalty. Default 3.");

// PWO-5 gate 3 (registered in docs/PWO5_PILOT_REGISTRATION.md Appendix A.1;
// name, type and default are frozen).
ABSL_FLAG(uint64_t, head_init_constant, 20260800,
          "PWO-5 section 7.2: kHeadInitConstant. NOT a run seed -- it is a "
          "fixed registered constant so that the initial head parameters are "
          "byte-identical across ALL SIX arms. Deriving from the triplet seed "
          "would give T1/P1/H1 one set and T2/P2/H2 another. Outside the "
          "reserved final-gate base-seed range by section 10.2's arithmetic.");
