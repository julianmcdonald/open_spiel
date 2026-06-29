# Swordmaster Planner Curriculum Plan

## Problem statement

The Stage 9 checkpoint is strong overall but almost never buys Swordmaster. When Beast is forced through an early `Smuggling -> Smuggling -> Shipping Recall -> Dividends -> Swordmaster` opening, the policy collapses rather than converting the third agent into advantage.

This is two separate learning failures:

1. **Acquisition failure**: the model does not consistently solve the early Solari/Swordmaster race. Early Swordmaster is not one action; it is a planning problem involving leader, seat order, hand access, Smuggling availability, shipping, Solari sources, opponent blocks, and queue position.
2. **Continuation failure**: when the model does get Swordmaster early, it has little practice playing the resulting three-agent game.

The plan is to hardcode the **goal**, not the route:

```text
Goal: legally own Swordmaster by a target deadline.
Do not enumerate every path to 8 Solari.
Use the game engine to search legal actions.
```

No free Swordmaster. No fake Solari. No raw Solari breadcrumbs.

## Design principles

- **Legal states only**: every curriculum state must be reached through engine-legal actions.
- **No route enumeration**: the planner may know the target action and success predicate, but not every recipe for reaching it.
- **Separate usage from acquisition**: first teach the post-Swordmaster continuation, then teach the model to race for Swordmaster.
- **Avoid seat-2 overfitting**: curriculum owners should rotate across P0/P1/P2/P3.
- **Avoid permanent Beast/Leto overfitting**: Beast/Leto are acceptable as reliable early-Swordmaster state generators, but the broader acquisition coach must train all seats/leaders.
- **Keep rewards clean**: use Solari and queue heuristics inside the planner only, not as PPO reward breadcrumbs.
- **Measure completion distribution**: track successes by seat, leader, round, and route pressure so the curriculum does not silently collapse to one narrow pattern.

## Core implementation pieces

The first implementation should not start with the full rollout planner. Stage 10A discards the acquisition prefix, so the route does not need to be strategically optimal. It only needs to be legal and to reach `HasSwordmaster(owner)`.

Build the pieces in this order:

1. **Navigation helper**: `FindActionOrCardPathToSpace(state, owner, target_space)`. This is needed immediately because even scripted warm-starts must choose a legal access card before placing on Smuggling/Swordmaster.
2. **Scripted legal warm-starts**: Beast/Leto lines that legally acquire early Swordmaster, aborting if any required step is illegal.
3. **Post-Swordmaster PPO continuation gym**: train from legal post-SM states using real GAE/Monte-Carlo returns. This is the stage that can correct the value head.
4. **Targeted post-SM search labels**: after the value head is less pessimistic, run IS-MCTS at generated post-SM states and distill policy improvements.
5. **Full acquisition planner/coach**: only then build `ChooseSwordmasterPlannerAction` / `RolloutSwordmasterRace` for arbitrary leaders/states.

This keeps the first implementation small and validates the highest-value hypothesis quickly: whether the model can learn to use Swordmaster once it sees enough legal post-Swordmaster states.

## Stage 10A navigation helper

The warm-start script and later planner both need a reliable way to select a card that legally reaches a target space. This helper must not select the first legal `SelectAgentCard` blindly. That was a weakness in the evaluation script.

Useful constants:

- `kActionAgentSpaceSwordmaster = 610`;
- `kActionAgentSpaceSmuggling = 614`;
- `kActionShippingRecall = 61`;
- `kActionShippingLevel1Dividends = 515`;
- `kActionSelectAgentCard0 = 1800`.

Helper behavior:

1. If the target space action is directly legal, return it.
2. If legal actions include `SelectAgentCard` actions, clone the state for each candidate card.
3. Apply the candidate card action in the clone.
4. Drain only forced/chance/singleton nodes if necessary.
5. Check whether the target space is legal in the resulting clone.
6. Among compatible cards, choose by planner score or current policy prior.

This allows scripted warm-starts to remain legal without hardcoding card-specific route logic.

## Later component: generic Swordmaster acquisition planner

The full planner is for Stage 10C and targeted acquisition label generation. It should run inside training/evaluation code and choose legal actions that maximize the chance of the chosen player owning Swordmaster before a deadline.

The planner should hardcode the goal, not the route:

- success predicate: `dune_state->HasSwordmaster(owner)`;
- target action: `kActionAgentSpaceSwordmaster`;
- do not enumerate every way to reach 8/7 Solari.

Everything else should be discovered by cloning states and applying legal actions.

### Planner API sketch

```cpp
struct SwordmasterPlannerConfig {
  int deadline_round = 3;
  int max_depth = 24;
  int rollouts_per_action = 8;
  double success_score = 10000.0;
  double round_bonus = 500.0;
  double solari_deficit_weight = 80.0;
  double current_policy_prior_weight = 0.10;
  bool use_policy_for_opponents = true;
  bool allow_fast_path_targets = true;
};

struct SwordmasterPlannerResult {
  Action action = kInvalidAction;
  double score = -std::numeric_limits<double>::infinity();
  bool immediate_swordmaster = false;
  bool target_space_path = false;
  bool found_successful_line = false;
  std::string reason;
};

SwordmasterPlannerResult ChooseSwordmasterPlannerAction(
    const State& state,
    Player owner,
    const SwordmasterPlannerConfig& cfg,
    std::mt19937* rng,
    BatchedEvaluator* evaluator);
```

### Planner behavior

At each owner decision before Swordmaster is owned:

1. If Swordmaster can be reached immediately, choose the legal path to it.
2. Otherwise, score all legal actions by short engine rollouts to the deadline.
3. Pick the action with highest expected Swordmaster-race score.
4. Replan at the next decision point.

Pseudo-code:

```cpp
Action ChooseSwordmasterPlannerAction(State& state, Player owner) {
  if (HasSwordmaster(state, owner)) return kInvalidAction;

  if (auto direct = FindActionOrCardPathToSpace(
          state, owner, kActionAgentSpaceSwordmaster)) {
    return direct.action;
  }

  Action best = kInvalidAction;
  double best_score = -inf;

  for (Action a : state.LegalActions()) {
    double score = 0.0;
    for (int i = 0; i < cfg.rollouts_per_action; ++i) {
      auto clone = state.Clone();
      clone->ApplyAction(a);
      score += RolloutSwordmasterRace(*clone, owner, cfg, rng);
    }
    score /= cfg.rollouts_per_action;

    // Small tie-break only. Do not let the current weak prior dominate.
    score += cfg.current_policy_prior_weight * PolicyLogProb(state, owner, a);

    if (score > best_score) {
      best_score = score;
      best = a;
    }
  }

  return best;
}
```

### Rollout scoring

The planner's rollout score is not game reward. It is a private heuristic used only to choose curriculum/coached actions.

Score priority:

1. `HasSwordmaster(owner)` before or at deadline: very large positive score.
2. Earlier purchase is better than later purchase.
3. If no purchase by deadline, score partial progress:
   - lower Solari deficit to Swordmaster cost;
   - useful shipping progress/recall position;
   - current or near-future access to Swordmaster/Smuggling;
   - preserving agent actions needed to complete the purchase;
   - avoiding being blocked this round.
4. Penalize losing the queue:
   - another player buys Swordmaster first;
   - Swordmaster/Smuggling is occupied before owner can use it;
   - owner reveals before resolving the purchase path.

The exact weights are tunable and should be logged. Because this score is not PPO reward, imperfect weights are less dangerous than Solari breadcrumbs.

### Opponent modeling during planner rollouts

Start simple:

- owner actions: planner/goal-biased policy;
- opponent actions: current policy;
- chance nodes: sampled normally.

Later, add a block-aware option:

- if an opponent can block Smuggling or Swordmaster and the owner is close to buying, the simulated opponent sometimes takes the block.

This matters because human play treats the Swordmaster queue as strategic. The planner should not learn only from cooperative worlds where nobody contests Smuggling.

## Training integration

The training should happen in phases. The phases are deliberately separated so bad post-Swordmaster play does not poison acquisition learning.

Revised ordering:

1. **10A**: scripted legal warm-start generator + post-Swordmaster continuation practice.
2. **10B**: targeted post-Swordmaster search labels and policy distillation, after 10A has improved the value head.
3. **10C**: generic acquisition planner/coach across all seats and leaders.
4. **10D**: consolidation with scaffolding removed.

Important technical constraint: the current search-label distillation path is policy-only. `SearchLabel` stores teacher action probabilities, not value targets, and the distillation loss is KL on logits. It does not directly train the value head.

That means Stage 10A must come first. The value head can learn that legal post-Swordmaster states are good only from PPO rollouts with real GAE/Monte-Carlo returns out of those states. Once 10A has de-biased the value head, Stage 10B can use IS-MCTS at post-SM roots to sharpen the policy head. If 10B runs too early, the search labels themselves may be degraded because PUCT leaf evaluation still bootstraps from a pessimistic value net.

Pre-Swordmaster search will also undervalue acquisition while the value net thinks post-Swordmaster states are bad. So the order is strict: first usage/value via PPO warm-starts, then post-SM policy distillation, then broad acquisition learning.

## Stage 10A: legal post-Swordmaster continuation gym

Purpose: teach the model how to play after early Swordmaster.

This stage uses legal warm-starts.

Stage 10A should **not** build or depend on the full rollout planner. Because the prefix is discarded, route quality is secondary. The implementation can use a small set of legal scripted warm-starts plus the card-to-space navigation helper:

- Beast/Leto R2 line: `Smuggling -> Smuggling -> ShippingRecall -> ShippingLevel1Dividends -> Swordmaster`.
- Abort if any script step is illegal or blocked.
- Prefer the policy-best compatible card when a script step requires `SelectAgentCard`.
- Start recording only after `HasSwordmaster(owner)` becomes true.

This deliberately avoids overbuilding the hardest part before validating the continuation-training hypothesis.

### Rollout flow

For a fraction of games:

1. Start a normal game.
2. Pick a curriculum owner with seat rotation across P0/P1/P2/P3.
3. Prefer Beast/Leto at first because they reliably generate legal R2 Swordmaster states.
4. Use the scripted legal warm-start line, plus `FindActionOrCardPathToSpace`, to acquire Swordmaster.
5. If blocked or failed before deadline, discard the warm-start attempt or fall back to normal rollout.
6. Once `HasSwordmaster(owner)` is true, clear the temporary prefix trajectory and start recording PPO transitions from that state.
7. Finish the game normally with the current policy.

Important: the forced prefix is not trained. It is just a legal state generator.

This is cleaner than adding per-transition policy weights for the forced prefix. We do not need to teach the model the Beast/Leto script yet; we need concentrated practice from legal post-Swordmaster states.

### Warm-start leader handling

Stage 10A may force the leader offer/draft branch for the warm-start owner, but only inside the unrecorded prefix.

Implementation options:

1. **Strict legal-branch forcing**: at leader-offer chance, choose an outcome containing Beast or Leto when one is available, then draft that leader for the rotating owner seat. Other seats draft normally or avoid the reserved leader.
2. **Offer-filtered attempts**: do not force the chance offer; activate the warm-start only when the rotating owner naturally has Beast/Leto available.

The first option generates more post-Swordmaster practice quickly. It is still an engine-legal state path, but not a naturally sampled leader distribution. That is acceptable for Stage 10A because the prefix is not trained and the goal is continuation practice, not acquisition policy learning.

The owner seat must still rotate. Do not hardcode P2.

Stage 10C removes this leader restriction by using the generic acquisition coach from normal starts across all leaders.

### Warm-start route diversity

Seat rotation alone is not enough. If every warm-start is the same Beast-Smuggling board texture, the continuation policy may not transfer cleanly to Swordmaster states reached by other leaders, different hands, combat Solari, Arrakeen lines, or R3 purchases.

Use an incremental route mix:

1. **MVP route**: Beast/Leto R2 Smuggling line.
2. **R2-R3 script window**: allow deadline round 3 so blocked or slower legal lines can still generate useful post-SM states.
3. **Natural opportunistic capture**: during normal rollouts, whenever any player legally buys Swordmaster by R3/R4, oversample or save the post-purchase continuation.
4. **Planner-assisted diversity**: once the full planner exists, add all-leader routes discovered by search.

The first route gets the gym working. The later routes broaden the post-SM distribution before acquisition training depends on it.

### Suggested flags

```bash
SM_WARMSTART_PROB=0.15
SM_WARMSTART_RECORD_PREFIX=false
SM_WARMSTART_OWNER_MODE=rotate
SM_WARMSTART_LEADERS=beast,leto
SM_WARMSTART_DEADLINE_ROUNDS=2,3
SM_WARMSTART_ROUTE_MIX=beast_leto_r2_script,natural_r3_capture
SM_WARMSTART_MAX_STEPS=250
SM_WARMSTART_SCRIPT_ONLY=true
```

### Stage 10A training command sketch

```bash
cp dune_ppo_stage_9_search_model.pt dune_ppo_stage_10a_sm_usage_model.pt
cp dune_ppo_stage_9_search_optimizer.pt dune_ppo_stage_10a_sm_usage_optimizer.pt

TOTAL_UPDATES=600 \
LEARNING_RATE=0.00001 \
ENTROPY_COEF=0.015 \
ANNEAL_LR=false \
GAE_LAMBDA=1.0 \
TARGET_KL=0.0 \
TLEILAXU_BREADCRUMB_WEIGHT=0.0 \
TLEILAXU_LEVEL7_BREADCRUMB_WEIGHT=0.0 \
SWORDMASTER_BREADCRUMB_WEIGHT=0.0 \
SM_WARMSTART_PROB=0.15 \
SM_WARMSTART_RECORD_PREFIX=false \
SM_WARMSTART_OWNER_MODE=rotate \
SM_WARMSTART_LEADERS=beast,leto \
SM_WARMSTART_DEADLINE_ROUNDS=2,3 \
CHECKPOINT_INTERVAL=100 \
SEED=10 \
RUN_PREFIX=dune_ppo_stage_10a_sm_usage \
MODEL_CHECKPOINT=dune_ppo_stage_10a_sm_usage_model.pt \
OPTIM_CHECKPOINT=dune_ppo_stage_10a_sm_usage_optimizer.pt \
stdbuf -oL -eL scripts/run_ppo_train.sh \
  2>&1 | tee training_log_ppo_stage_10a_sm_usage.txt
```

### Stage 10A success criteria

- Conditional forced/warm-start Beast/Leto early-Swordmaster continuation improves materially from the current 13% Beast result.
- Value estimates for legal post-Swordmaster states stop being catastrophically pessimistic, measured against realized returns from warm-start continuations.
- Third-agent utilization improves:
  - fewer reveal/end-turn decisions with agents remaining;
  - more useful third placements;
  - better VP/return after legal early Swordmaster.
- Normal Stage 9 head-to-head does not collapse, but it is not expected to move much yet.

Natural Swordmaster acquisition and headline winrate are not the main gates for this stage. Stage 10A trains states the current policy rarely reaches on its own, so a flat headline metric is not failure. The correct gate is conditional performance from legal post-Swordmaster states.

## Stage 10B: targeted post-Swordmaster search-label distillation

Purpose: bake good Swordmaster continuation decisions into the policy head using the mechanism that already produced the Stage 9 champion.

This should run after Stage 10A has improved conditional post-Swordmaster value/returns. Search labels can sharpen the action distribution, but the current label format cannot provide a value target.

### Label-generation approach

Use the warm-start generator to produce legal post-Swordmaster roots:

1. Start normal games.
2. Use scripted legal warm-starts to reach `HasSwordmaster(owner)`.
3. At the immediate post-purchase state and subsequent third-agent decision states, run IS-MCTS.
4. Write labels only for target states:
   - immediately after Swordmaster purchase;
   - third-agent decisions after early Swordmaster;
   - reveal/end-turn decisions where agents remain;
   - high-leverage R3-R5 placement decisions with three agents.

Use a higher search budget than broad Stage 9 labels for this sparse branch.

Suggested first pass:

```bash
--target_labels=8192
--max_simulations=200
--target_teacher_kl=0.05
--search_fraction=0.30
--sm_label_mode=post_purchase_usage
--sm_warmstart_generator=beast_leto_script
```

Then distill conservatively:

```bash
SEARCH_LABEL_DIR=search_labels/stage10_sm_usage_s200_kl005
SEARCH_LAMBDA=0.05
SEARCH_MINIBATCHES_PER_UPDATE=1
SEARCH_MINIBATCH_SIZE=512
```

### Stage 10B success criteria

- Conditional post-Swordmaster return improves.
- Third-agent utilization improves under search-distilled policy.
- Policy choices at third-agent and post-SM placement decisions improve.
- Normal head-to-head remains stable.

## Stage 10C: generic Swordmaster acquisition coach

Purpose: teach the model to solve the Swordmaster race from normal starts across all seats and leaders.

This stage uses action influence, not hard forcing.

This is optional/skippable until 10A/10B have improved post-Swordmaster value. If the value net still thinks post-SM states are bad, a pre-SM planner/search coach will often fail to value the acquisition because the long-horizon payoff is bootstrapped through a bad leaf evaluation.

### Behavior policy

During early rounds, with some probability, replace normal policy sampling with a goal-biased behavior policy:

```text
behavior_logits[a] = model_logits[a] + beta * normalized_planner_score[a]
```

Then sample from `behavior_logits`.

This means the action is still legal and still chosen from a distribution. The old log-prob stored for PPO must be the behavior-policy log-prob, not the unmodified model log-prob.

The planner supplies `planner_score[a]` by cloning the state, applying `a`, and rolling/scoring toward the Swordmaster deadline.

### Why behavior-logit influence instead of Solari breadcrumbs?

Solari breadcrumbs distort the game objective. They teach hoarding and may interfere with High Council, TSMF, tech, combat, and other valid Solari uses.

The coach uses Solari only as a private planning feature. PPO still receives game returns and existing VP-shaped rewards.

### Suggested flags

```bash
SM_COACH_PROB=0.20
SM_COACH_BETA=1.5
SM_COACH_ROUNDS=1,2,3,4
SM_COACH_DEADLINE_ROUND=4
SM_COACH_OWNER_MODE=all_current_players
SM_PLANNER_ROLLOUTS_PER_ACTION=4
SM_PLANNER_MAX_DEPTH=24
SM_PLANNER_USE_POLICY_OPPONENTS=true
SM_PLANNER_BLOCK_AWARE_OPPONENTS=false
```

Start with modest `beta`. Increase only if the model still never reaches early-Swordmaster states.

### Stage 10C training command sketch

```bash
cp dune_ppo_stage_10b_sm_usage_distill_model.pt dune_ppo_stage_10c_sm_acquire_model.pt
cp dune_ppo_stage_10b_sm_usage_distill_optimizer.pt dune_ppo_stage_10c_sm_acquire_optimizer.pt

TOTAL_UPDATES=1000 \
LEARNING_RATE=0.00001 \
ENTROPY_COEF=0.015 \
ANNEAL_LR=false \
GAE_LAMBDA=1.0 \
TARGET_KL=0.0 \
TLEILAXU_BREADCRUMB_WEIGHT=0.0 \
TLEILAXU_LEVEL7_BREADCRUMB_WEIGHT=0.0 \
SWORDMASTER_BREADCRUMB_WEIGHT=0.0 \
SM_COACH_PROB=0.20 \
SM_COACH_BETA=1.5 \
SM_COACH_DEADLINE_ROUND=4 \
CHECKPOINT_INTERVAL=100 \
SEED=11 \
RUN_PREFIX=dune_ppo_stage_10c_sm_acquire \
MODEL_CHECKPOINT=dune_ppo_stage_10c_sm_acquire_model.pt \
OPTIM_CHECKPOINT=dune_ppo_stage_10c_sm_acquire_optimizer.pt \
stdbuf -oL -eL scripts/run_ppo_train.sh \
  2>&1 | tee training_log_ppo_stage_10c_sm_acquire.txt
```

### Stage 10C success criteria

- Natural Swordmaster acquisition rate rises in self-play.
- Acquisition is not seat-2 concentrated.
- Acquisition is not Beast/Leto-only.
- Forced early-Swordmaster continuation remains improved.
- Normal head-to-head versus Stage 9 stays competitive or improves.

## Stage 10C-2: targeted acquisition search-label distillation

Purpose: bake good Swordmaster race decisions into the policy head using search labels in the sparse pre-purchase subgame.

Do this only after post-Swordmaster usage/value has improved. Stage 9 generated broad search labels and improved strength, but the label distribution barely covered Swordmaster states. Acquisition labels should target:

- Smuggling-contested states;
- states where owner is close to 8/7 Solari;
- states where Swordmaster is legal or one action away;
- opponent decisions where blocking Smuggling/Swordmaster is relevant.

### Label-generation approach

Modify or wrap `dune_search_teacher` so it can use the planner as a state-distribution generator:

1. Start normal games.
2. Use the acquisition planner/coach to push some games into Swordmaster-race states.
3. Run IS-MCTS at selected strategic states.
4. Write labels only for states matching target filters.

Use a higher search budget than broad Stage 9 labels for this sparse tactical branch.

Suggested first pass:

```bash
--target_labels=8192
--max_simulations=200
--target_teacher_kl=0.05
--search_fraction=0.30
--sm_label_mode=race_and_blocking
```

Then distill conservatively:

```bash
SEARCH_LABEL_DIR=search_labels/stage10_sm_acquisition_s200_kl005
SEARCH_LAMBDA=0.05
SEARCH_MINIBATCHES_PER_UPDATE=1
SEARCH_MINIBATCH_SIZE=512
```

## Stage 10D: consolidation

Purpose: remove the scaffolding and check whether Swordmaster persists under normal play.

Run normal PPO/search-distillation without warm-starts or coach:

- `SM_WARMSTART_PROB=0.0`;
- `SM_COACH_PROB=0.0`;
- `SWORDMASTER_BREADCRUMB_WEIGHT=0.0`.

If Swordmaster is genuinely beneficial and the model now understands the branch, acquisition should remain above Stage 9's near-zero rate without artificial pressure.

## Implementation checklist

### 1. Planner helper

Add a helper in C++ first, either embedded in `dune_ppo_train.cc` for rapid iteration or split into reusable files once stable:

- `ContainsAction(actions, action)`;
- `IsSelectAgentCardAction(action)`;
- `DrainForcedNodesForPlanner(clone)`;
- `FindActionOrCardPathToSpace(state, owner, target_space)`;
- `ScoreSwordmasterRaceState(state, owner, cfg)`;
- `RolloutSwordmasterRace(state, owner, cfg, rng, evaluator)`;
- `ChooseSwordmasterPlannerAction(state, owner, cfg, rng, evaluator)`.

### 2. Warm-start rollout mode

Modify PPO rollout collection to support:

- normal rollout from initial state;
- warm-start prelude from initial state to post-Swordmaster;
- clearing/discarding prefix transitions before recording;
- per-game stats for attempts/success/failure reasons.

Do not add policy weights unless later needed. For Stage 10A, not recording the prefix is simpler and safer.

### 3. Coach behavior policy

Add optional early-round coach influence:

- compute planner scores for legal actions;
- normalize scores;
- add `SM_COACH_BETA * score` to legal logits;
- sample action;
- store old log-prob from the coached behavior distribution.

This is more invasive than warm-starts and should come after Stage 10A/10B.

### 4. Script passthroughs

Add environment variable passthroughs in `scripts/run_ppo_train.sh` for:

- `SM_WARMSTART_PROB`;
- `SM_WARMSTART_RECORD_PREFIX`;
- `SM_WARMSTART_OWNER_MODE`;
- `SM_WARMSTART_LEADERS`;
- `SM_WARMSTART_DEADLINE_ROUNDS`;
- `SM_WARMSTART_ROUTE_MIX`;
- `SM_WARMSTART_SCRIPT_ONLY`;
- `SM_COACH_PROB`;
- `SM_COACH_BETA`;
- `SM_COACH_DEADLINE_ROUND`;
- `SM_PLANNER_ROLLOUTS_PER_ACTION`;
- `SM_PLANNER_MAX_DEPTH`.

### 5. Diagnostics

Log per update:

- warm-start attempts;
- warm-start completions;
- warm-start abandonment reasons;
- completions by seat;
- completions by leader;
- purchase round distribution;
- number of post-Swordmaster transitions collected;
- coached-action count;
- coached-action average planner score;
- natural Swordmaster purchases in rollout;
- reveal/end-turn with agents remaining after Swordmaster.

Without these diagnostics, the curriculum could silently overfit one seat/leader/path.

## Evaluation plan

### Required evaluations

1. **Forced Beast/Leto early-Swordmaster continuation**
   - Same legal early-Swordmaster setup.
   - Score by engine returns, not just raw VP ties.
   - Track third-agent utilization.

2. **Natural self-play behavior**
   - Swordmaster acquisitions/game.
   - Acquisition by seat and leader.
   - Acquisition round distribution.
   - High Council / TSMF / tech rates to detect distortions.

3. **Head-to-head strength**
   - Stage 10 checkpoint vs Stage 9 champion.
   - Stage 10 checkpoint vs Stage 8/7 baselines if needed.

4. **Matched branch diagnostic**
   - Clone exact states before Swordmaster purchase.
   - Branch A: buy Swordmaster.
   - Branch B: policy/no-Swordmaster alternative.
   - Continue both with matched seeds/chance where possible.

### Gating targets

Early Stage 10A:

- forced early-Swordmaster continuation improves materially;
- no catastrophic head-to-head collapse.

Stage 10B:

- conditional post-Swordmaster action quality improves further after search distillation;
- third-agent utilization improves further;
- normal winrate remains stable even if acquisition has not yet increased.

Stage 10C:

- natural Swordmaster acquisition rises above Stage 9's 0.02/game;
- not concentrated only in one seat;
- not exclusively Beast/Leto;
- normal winrate remains competitive.

Stage 10D:

- Swordmaster acquisition persists with coach disabled;
- Stage 10 beats or matches Stage 9 in large head-to-head validation.

## Risks and mitigations

### Risk: Beast/Leto overfitting

Mitigation:

- rotate owner seats in Stage 10A;
- do not record forced prefix;
- add generic all-leader coach in Stage 10C after usage/value improves;
- log completion distribution.

### Risk: planner finds unrealistic routes because opponents do not block

Mitigation:

- use current policy opponents first;
- add block-aware opponent simulation if needed;
- include opponent block decisions in targeted search labels.

### Risk: coach destabilizes PPO

Mitigation:

- Stage 10A uses warm-starts without changing behavior log-probs;
- Stage 10C starts with small `SM_COACH_BETA`;
- store behavior-policy log-probs correctly when coach is active;
- compare against a no-coach control checkpoint.

### Risk: Solari heuristic distorts play

Mitigation:

- keep Solari inside planner scoring only;
- do not add Solari PPO rewards;
- remove coach in consolidation.

## Bottom line

The curriculum should not try to list every way to get 8 Solari. That is the strategic problem humans are solving.

Instead:

1. Build the navigation helper and scripted Beast/Leto warm-starts first.
2. Run Stage 10A PPO from legal post-Swordmaster states, rotating seats and not recording the forced prefix. This is the value-head correction stage.
3. After 10A improves conditional post-SM value/returns, generate targeted post-SM IS-MCTS labels and distill policy improvements in Stage 10B.
4. Only then build the generic engine-backed acquisition planner/coach for all seats/leaders in Stage 10C.
5.  - Optionally add targeted acquisition search labels once post-SM value is trustworthy.
  - Remove scaffolding and verify the behavior persists in normal self-play.

## Execution History & Results (Stages 10A to 10D Probe)

### Stage 10A: Legal Swordmaster Warm-Start Value Correction
- **Objective**: Correct the value head of the PPO model in legal post-Swordmaster states using seat-rotating warm-starts (Beast and Leto).
- **Configuration**:
  - `TOTAL_UPDATES=600`
  - `SM_WARMSTART_PROB=0.15`
  - `SM_WARMSTART_RECORD_PREFIX=false`
  - `SM_WARMSTART_OWNER_MODE=rotate`
  - `SM_WARMSTART_LEADERS=beast,leto`
  - `LEARNING_RATE=0.00001`, `ENTROPY_COEF=0.015`, `GAE_LAMBDA=1.0`
- **Outcomes**:
  - Forced Beast early-SM P2 winrate improved from **13.52%** to **13.94%** (Average VP increased from 7.65 to 8.03; average return improved from -0.61 to -0.52).
  - Wasted reveals (revealing with 1+ unused agents after SM) dropped from **5.4%** to **2.4%**.
  - Normal-play stability was maintained with balanced seat win rates.

### Stage 10B: Targeted Post-Swordmaster Search-Label Distillation
- **Objective**: Use IS-MCTS at post-Swordmaster states to generate search labels and distill them into the policy head via KL divergence.
- **Outcomes**:
  - Sharpened the policy's action distribution in post-SM states (such as third-agent placement and reveal decisions).
  - Maintained baseline win rates in unconstrained play.

### Stage 10C: Generic Swordmaster Acquisition Planner/Coach
- **Objective**: Teach the policy to solve the pre-SM race from normal starts across all seats and leaders using behavior policy logit-coercion.
- **Configuration**:
  - `TOTAL_UPDATES=1000`
  - `SM_COACH_PROB=0.20`, `SM_COACH_BETA=1.5`
  - `SM_COACH_DEADLINE_ROUND=4`
  - `SWORDMASTER_BREADCRUMB_WEIGHT=0.0`, `SM_WARMSTART_PROB=0.0`
- **Outcomes (Stage 10C-3 Model)**:
  - **Natural SM rate** rose dramatically to **0.81 acquisitions/game** in unconstrained self-play (compared to ~0.02 in Stage 9 baseline).
  - Wasted reveals in unconstrained play remained low (96.5% fully utilized).
  - Forced Beast early-SM winrate stabilized at **13.02%** (VP = 7.97, return = -0.7969).
  - Unconstrained play headline win shares were stable and balanced (P0: 23.5%, P1: 32.0%, P2: 26.5%, P3: 18.0%).

### Stage 10D Probe: Scaffold-Off Consolidation Probe (Full Stage 10D Gated Out)
- **Objective**: Test if the learned Swordmaster acquisition and usage behaviors persist in pure self-play when all training scaffolding is removed. This probe gates whether we proceed to the full Stage 10D consolidation stage.
- **Configuration**:
  - Initialized from Stage 10C-3 coached model.
  - `TOTAL_UPDATES=100` (terminated early at Update 137).
  - `SM_WARMSTART_PROB=0.0`, `SM_COACH_PROB=0.0`, `SWORDMASTER_BREADCRUMB_WEIGHT=0.0`, `SEARCH_LAMBDA=0.0`.
- **Outcomes (At Update 100 Checkpoint)**:
  - **SM/game collapsed completely** from `0.81` down to **0.0667** acquisitions/game (Total of 20 acquisitions across 300 evaluated games). In self-play stochastic rollouts, the acquisition rate decayed to `~1% to 3%` by Update 137.
  - Forced Beast early-SM P2 winrate dropped to **9.32%** (below the `>= 13.0%` gate).
  - General gameplay win shares and VP remained stable, but the tactical acquisition behavior was completely scaffold-dependent.
- **Decision**: The consolidation probe failed. The full Stage 10D consolidation run was gated out and not executed. Training was terminated early.
- **Next Steps Recommendation**: Go back to a coaching or curriculum pass that explicitly addresses early round (R3-R5) Solari conversion values and increases the post-purchase usage advantage, ensuring the model internally values the extra agent.

