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

## Core component: Swordmaster acquisition planner

Build a small goal-seeking planner that runs inside training/evaluation code and chooses legal actions that maximize the chance of the chosen player owning Swordmaster before a deadline.

The planner should not know "Smuggling twice" as a hardcoded route. It should know:

- success predicate: `dune_state->HasSwordmaster(owner)`;
- target action: `kActionAgentSpaceSwordmaster`;
- useful constants for common direct actions:
  - `kActionAgentSpaceSwordmaster = 610`;
  - `kActionAgentSpaceSmuggling = 614`;
  - `kActionShippingRecall = 61`;
  - `kActionShippingLevel1Dividends = 515`;
  - `kActionSelectAgentCard0 = 1800`.

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

### Finding a legal path to a target space

The planner must not select the first legal `SelectAgentCard` blindly. That was a weakness in the evaluation script.

Instead:

1. If the target space action is directly legal, return it.
2. If legal actions include `SelectAgentCard` actions, clone the state for each candidate card.
3. Apply the candidate card action in the clone.
4. Drain only forced/chance/singleton nodes if necessary.
5. Check whether the target space is legal in the resulting clone.
6. Among compatible cards, choose by planner score or current policy prior.

This allows the planner to find legal access without hardcoding card-specific route logic.

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

## Stage 10A: legal post-Swordmaster continuation gym

Purpose: teach the model how to play after early Swordmaster.

This stage uses legal warm-starts.

### Rollout flow

For a fraction of games:

1. Start a normal game.
2. Pick a curriculum owner with seat rotation across P0/P1/P2/P3.
3. Prefer Beast/Leto at first because they reliably generate legal R2 Swordmaster states.
4. Use the Swordmaster planner/fast path to legally acquire Swordmaster.
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

Stage 10B removes this leader restriction by using the generic acquisition coach from normal starts across all leaders.

### Suggested flags

```bash
SM_WARMSTART_PROB=0.15
SM_WARMSTART_RECORD_PREFIX=false
SM_WARMSTART_OWNER_MODE=rotate
SM_WARMSTART_LEADERS=beast,leto
SM_WARMSTART_DEADLINE_ROUND=2
SM_WARMSTART_MAX_STEPS=250
SM_PLANNER_ROLLOUTS_PER_ACTION=4
SM_PLANNER_MAX_DEPTH=20
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
SM_WARMSTART_DEADLINE_ROUND=2 \
CHECKPOINT_INTERVAL=100 \
SEED=10 \
RUN_PREFIX=dune_ppo_stage_10a_sm_usage \
MODEL_CHECKPOINT=dune_ppo_stage_10a_sm_usage_model.pt \
OPTIM_CHECKPOINT=dune_ppo_stage_10a_sm_usage_optimizer.pt \
stdbuf -oL -eL scripts/run_ppo_train.sh \
  2>&1 | tee training_log_ppo_stage_10a_sm_usage.txt
```

### Stage 10A success criteria

- Forced Beast/Leto early-Swordmaster continuation improves materially from the current 13% Beast result.
- Third-agent utilization improves:
  - fewer reveal/end-turn decisions with agents remaining;
  - more useful third placements;
  - better VP/return after legal early Swordmaster.
- Normal Stage 9 head-to-head does not collapse.

Natural Swordmaster acquisition is not the main gate for this stage.

## Stage 10B: generic Swordmaster acquisition coach

Purpose: teach the model to solve the Swordmaster race from normal starts across all seats and leaders.

This stage uses action influence, not hard forcing.

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

### Stage 10B training command sketch

```bash
cp dune_ppo_stage_10a_sm_usage_model.pt dune_ppo_stage_10b_sm_acquire_model.pt
cp dune_ppo_stage_10a_sm_usage_optimizer.pt dune_ppo_stage_10b_sm_acquire_optimizer.pt

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
RUN_PREFIX=dune_ppo_stage_10b_sm_acquire \
MODEL_CHECKPOINT=dune_ppo_stage_10b_sm_acquire_model.pt \
OPTIM_CHECKPOINT=dune_ppo_stage_10b_sm_acquire_optimizer.pt \
stdbuf -oL -eL scripts/run_ppo_train.sh \
  2>&1 | tee training_log_ppo_stage_10b_sm_acquire.txt
```

### Stage 10B success criteria

- Natural Swordmaster acquisition rate rises in self-play.
- Acquisition is not seat-2 concentrated.
- Acquisition is not Beast/Leto-only.
- Forced early-Swordmaster continuation remains improved.
- Normal head-to-head versus Stage 9 stays competitive or improves.

## Stage 10C: targeted search-label distillation

Purpose: bake good Swordmaster decisions into the policy head using search labels in the sparse subgame.

Stage 9 generated broad search labels and improved strength, but the label distribution barely covered Swordmaster states. Stage 10C should generate labels specifically around:

- Smuggling-contested states;
- states where owner is close to 8/7 Solari;
- states where Swordmaster is legal or one action away;
- immediately after Swordmaster purchase;
- third-agent decisions after early Swordmaster;
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
--sm_label_mode=race_and_post_purchase
```

Then distill conservatively:

```bash
SEARCH_LABEL_DIR=search_labels/stage10_sm_targeted_s200_kl005
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

This is more invasive than warm-starts and should come after Stage 10A.

### 4. Script passthroughs

Add environment variable passthroughs in `scripts/run_ppo_train.sh` for:

- `SM_WARMSTART_PROB`;
- `SM_WARMSTART_RECORD_PREFIX`;
- `SM_WARMSTART_OWNER_MODE`;
- `SM_WARMSTART_LEADERS`;
- `SM_WARMSTART_DEADLINE_ROUND`;
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
- quickly add generic all-leader coach in Stage 10B;
- log completion distribution.

### Risk: planner finds unrealistic routes because opponents do not block

Mitigation:

- use current policy opponents first;
- add block-aware opponent simulation if needed;
- include opponent block decisions in targeted search labels.

### Risk: coach destabilizes PPO

Mitigation:

- Stage 10A uses warm-starts without changing behavior log-probs;
- Stage 10B starts with small `SM_COACH_BETA`;
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

1. Use Beast/Leto as reliable legal early-Swordmaster state generators for post-Swordmaster practice, rotating seats and not recording the forced prefix.
2. Build a generic engine-backed Swordmaster planner that searches legal actions toward `HasSwordmaster(owner)` by deadline.
3. Use that planner first for warm-starts, then as a soft action coach across all seats/leaders.
4. Generate targeted search labels around the Swordmaster race and post-purchase decisions.
5. Remove scaffolding and verify the behavior persists in normal self-play.
