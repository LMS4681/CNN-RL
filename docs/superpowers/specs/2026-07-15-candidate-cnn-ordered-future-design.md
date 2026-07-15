# Candidate CNN and Ordered Future Design

Date: 2026-07-15
Status: Approved design

## 1. Objective

AllocRL must keep the existing incremental placement simulator and MaskablePPO
training flow while making the model reason about the long-term spatial effect of
placing the current block in each workspace.

The revised model must:

- use CNN features as evidence about the spatial state left after a candidate
  placement;
- provide the current block and the next `K` blocks to the policy in their actual
  decision order;
- keep exact immediate placeability as a separate structured feature;
- remove pointer attention, block self-attention, and PointNet-like complexity;
- preserve the final delay/dropout optimization objective;
- make the contribution of future information and CNN learning measurable through
  controlled ablations.

## 2. Scope and non-goals

In scope:

- observation construction;
- active-workspace selection;
- feature extractor architecture;
- timing of reward delivery;
- deterministic seeding and diagnostics;
- training, evaluation, visualization, ONNX, and regression-test compatibility.

Out of scope:

- changing simulator placement, delay, rotation, or dropout rules;
- replacing MaskablePPO;
- multi-step action rollouts or a learned world model;
- PointNet or point-cloud representations;
- attention of any kind;
- a hand-crafted fragmentation term in the primary reward;
- changing the hard-constraint action-mask semantics.

## 3. Design decisions

### 3.1 Immediate placeability remains structured

`placeable_now` remains in `ws_meta`. It is calculated by the same placement
strategy used by the simulator, but it is not a CNN classification target and is
not added to the action mask. A block may still be assigned to a currently full
workspace and wait for space to become available.

This separation gives each path one responsibility:

- `placeable_now`: exact immediate feasibility;
- CNN: spatial layout, remaining occupancy lifetime, and post-placement
  fragmentation evidence;
- PPO: long-term action value under current and upcoming demand.

### 3.2 Attention is removed

`PointerAttentionCnnExtractor` and `BlockSetAttentionCnnExtractor` are removed.
Future blocks are represented by fixed ordered slots and a normal MLP. Swapping
future slots must change the encoded representation.

### 3.3 Only active workspaces enter the model

The configured active workspace codes are resolved before environment creation.
The environment receives only those workspace objects, so observation and action
spaces contain `N_active` workspaces rather than all workspaces with inactive rows
filled with zero.

Workspace codes remain attached to each object and are used in logs and
visualization. Model action index `i` means index `i` in the filtered workspace
list.

## 4. Observation contract

With `N` active workspaces, grid size `G`, and future horizon `K`, the observation
is:

| Key | Shape | Meaning |
| --- | --- | --- |
| `block` | `(10,)` | Current block physical, temporal, and progress features |
| `future_blocks` | `(K, 8)` | Next decision blocks in exact decision order |
| `future_mask` | `(K,)` | `1` for a valid future slot, `0` for padding |
| `grids` | `(N, 4, G, G)` | Candidate-conditioned workspace images |
| `ws_meta` | `(N, 3)` | Scale, occupancy ratio, and `placeable_now` |

`K` remains configurable and defaults to `4`. All extractors support `K=0` for
ablation, but the recommended model uses `K=4`.

### 4.1 Grid channels

The four channels are:

1. current occupancy mask;
2. normalized remaining occupancy duration;
3. workspace boundary mask;
4. current block candidate-placement mask.

Channels 0-2 remain cacheable by workspace state and environment date. Channel 3
is created for the current decision and concatenated after fetching the cached base
grid.

For an immediately placeable candidate, channel 3 marks the exact rectangle and
orientation returned by the placement strategy. For a candidate that is not
immediately placeable, channel 3 is all zeros and `placeable_now` is zero. No
future position is fabricated. The remaining-duration channel provides the
evidence from which the policy can learn whether waiting is useful.

### 4.2 Candidate-position cache

The placeability calculation returns one record per workspace:

```text
CandidatePlacement(position_x, position_y, rotated, placeable)
```

The same records build both `placeable_now` and channel 3. Candidate search must
operate on a cloned block or otherwise guarantee restoration in a `finally` block.
Observation construction must not mutate block dimensions, orientation, position,
workspace contents, or simulator state.

The simulator remains authoritative and independently applies the selected action.
Tests enforce that the observation candidate and simulator placement agree.

## 5. Feature extractors

The supported extractor names are:

- `structured`;
- `fixed-grid`;
- `candidate-cnn`.

All three produce a standard SB3 feature vector and continue to use
`MultiInputPolicy` and MaskablePPO.

### 5.1 Ordered block context

All extractors share an ordered block encoder:

```text
current context = Linear(10, 64) -> ReLU -> Linear(64, 32) -> ReLU
future input    = flatten(future_blocks * future_mask) + future_mask
future context  = Linear(9*K, 128) -> ReLU -> Linear(128, 64) -> ReLU
block context   = concat(current context, future context) -> 96
```

Flattening fixed slots gives different learned weights to future position 1,
position 2, and so on. Padding values are zeroed before flattening, and the mask is
included so that padding is distinguishable from a real all-zero feature row.
When `K=0`, the future context is a constant zero vector of length 64 and no
future-encoder parameters are created.

### 5.2 CandidateCnnExtractor

The trainable CNN uses four input channels and batch-independent normalization:

```text
Conv2d(4, 32, kernel=5, stride=1, padding=2)
-> GroupNorm(8, 32) -> ReLU
Conv2d(32, 64, kernel=3, stride=2, padding=1)
-> GroupNorm(8, 64) -> ReLU
Conv2d(64, 64, kernel=3, stride=2, padding=1)
-> GroupNorm(8, 64) -> ReLU
AdaptiveAvgPool2d(8, 8)
-> flatten -> Linear(4096, 128) -> ReLU
```

Supported grid sizes are at least 32, so the convolution stack reaches at least
8x8 before adaptive pooling. The required invariants are four input channels,
GroupNorm rather than BatchNorm, and an 8x8 final spatial representation before
projection.

CNN weights are shared across workspaces and are trainable through the PPO actor
and critic losses.

### 5.3 FixedGridExtractor

Each four-channel image is adaptively downsampled to 8x8 with a non-learned pooling
operation and flattened. There are no convolution parameters. A shared learned MLP
may combine those fixed pixels with structured context; this experiment tests
whether convolution and a learned spatial representation add value beyond direct
image access.

### 5.4 StructuredExtractor

The image is ignored. The extractor uses only block context and `ws_meta`. It must
contain no convolution parameters.

### 5.5 Workspace fusion

For workspace `i`, the selected image representation is concatenated with
`ws_meta[i]` and the shared block context. A shared workspace-fusion MLP produces
one candidate embedding per workspace. Candidate embeddings are flattened in the
stable filtered-workspace order and projected to the policy feature dimension,
default `256`.

The exact fusion dimensions are:

```text
structured workspace input: 3 + 96 = 99
fixed-grid workspace input:  256 + 3 + 96 = 355
candidate-CNN input:         128 + 3 + 96 = 227

workspace fusion: Linear(input, 128) -> ReLU
                  -> Linear(128, 64) -> ReLU
global fusion:    flatten(N * 64) -> Linear(N * 64, features_dim) -> ReLU
```

No workspace attention, pointer score, or custom action head is introduced. The
existing SB3 policy/value MLPs produce action logits and values from this feature
vector.

## 6. Reward delivery

The delay/dropout score remains unchanged:

```text
score(delay <= 2) = +1.0
score(3 <= delay <= 7) = -(delay - 2) / 5
score(dropout) = -2.0
terminal_score = sum(score(block)) / total_block_count
```

Immediate placement success/failure shaping and interval partial-replay shaping
are removed. They are replaced by reward delivery when block outcomes become
final.

### 6.1 Resolved reward

The environment tracks block indices whose final result has already been emitted.
After each selected action and all automatic retries performed while advancing to
the next decision, it collects newly resolved blocks:

```text
resolved_reward_t = sum(score(newly_resolved_block)) / total_block_count
```

A block is newly resolved only when its final delay is known or it drops out. A
delayed retry with no final result emits no reward, and every block emits exactly
once.

Blocks auto-resolved as globally infeasible while advancing during `reset()` are
queued and emitted with the first subsequent `step()`. An environment with no
agent decision at all is rejected during construction with a clear data/config
error rather than returning an unstepable episode.

### 6.2 Terminal residual

At termination:

```text
terminal_residual = terminal_score - emitted_resolved_reward_sum
reward_t += terminal_residual
```

The invariant is:

```text
sum(all environment rewards in the episode) == terminal_score
```

within floating-point tolerance. `terminal_score` remains available in `info` for
evaluation and reporting even though most or all of it has already been delivered
incrementally.

### 6.3 PPO return settings

- `gamma` remains `1.0`;
- `gae_lambda` becomes an explicit CLI/config value, default `0.98`;
- `n_steps` remains at least two typical episode lengths.

No fragmentation reward is included in the initial implementation. If learning
remains insufficient after the approved ablation, an optional potential-based
future-optionality signal requires a separate design decision.

## 7. Training configuration and reproducibility

The CLI exposes:

```text
--extractor structured|fixed-grid|candidate-cnn
--n-future-blocks 4
--seed <integer>
--gae-lambda 0.98
```

Pointer/block attention choices and `embed_dim`/`num_heads` arguments are removed.

One seed controls:

- Python randomness used by constraints or helpers;
- NumPy;
- PyTorch;
- Gymnasium reset;
- SyntheticBlockGenerator;
- workspace layout variation;
- PPO initialization.

Evaluation uses persisted fixed synthetic episode specifications plus the original
CSV evaluation environment. Training episodes must not be reused as fixed
evaluation episodes.

## 8. CNN diagnostics

Training logs include:

- `cnn_gradient_norm`;
- `cnn_weight_change`;
- `workspace_feature_variance`;
- `candidate_channel_sensitivity`.

Gradient hooks accumulate CNN parameter gradient norms during PPO backward passes.
The callback reads and resets those accumulators at rollout boundaries.

Weight change is the norm between the current CNN parameter snapshot and the
snapshot from the previous completed PPO update.

Feature variance is measured across active workspace features from sampled rollout
observations. Candidate sensitivity is the mean feature-vector distance between an
observation and the same observation with only channel 3 zeroed. These are
diagnostics, not losses.

Structured and fixed-grid runs record the CNN-only fields as not applicable rather
than zero, so plots do not imply a dead CNN.

## 9. Ablation plan

| ID | Extractor | Future horizon | Question |
| --- | --- | --- | --- |
| A | `structured` | 0 | Minimum current-state baseline |
| B | `structured` | 4 | Does ordered future information help? |
| C | `fixed-grid` | 4 | Does direct image access help without CNN? |
| D | `candidate-cnn` | 0 | Does learned spatial state help alone? |
| E | `candidate-cnn` | 4 | Recommended complete model |

Screening uses three seeds. Final comparison uses at least five seeds with identical
training budgets and fixed evaluation episodes.

Primary metrics, in order, are:

1. mean terminal score;
2. dropout rate;
3. mean delay days and delayed-block count;
4. learning speed and variance across seeds;
5. future utilization of space left by earlier placements.

The fifth metric is the evaluation-only retained-choice ratio. For the next `K`
blocks, count valid workspace choices immediately before and immediately after the
selected placement, sum across the valid future slots, and report
`after / max(before, 1)`. It is not included in observations or reward.

The candidate CNN is justified when E improves over B by either an absolute mean
terminal score of at least `0.05` or a relative dropout reduction of at least
`10%`, and the direction of improvement is consistent in at least four of five
seeds. E versus C tests learned convolution; E versus D tests future information.

CNN pretraining is not part of the first implementation. If end-to-end CNN learning
fails diagnostics, a later optional pretraining target may predict the number of
workspace choices retained for the next K blocks after candidate placement. That
change requires separate approval.

## 10. File-level changes

### `AllocRL/alloc_env/alloc_env.py`

- change grid observation to four channels;
- compute/cache candidate placements;
- keep ordered future observations;
- emit newly resolved rewards and terminal residual;
- remove immediate and partial-replay shaping;
- expose reward components in `info`.

### `AllocRL/alloc_env/occupancy_grid.py`

- retain the cached three-channel base renderer;
- add candidate-mask rendering from a non-mutating candidate record;
- combine base and candidate channels for observation.

### `AllocRL/alloc_env/cnn_extractor.py`

- remove both attention extractors;
- replace BatchNorm CNN with GroupNorm candidate CNN;
- add ordered block context;
- add structured, fixed-grid, and candidate-CNN extractors.

### `AllocRL/alloc_env/incremental_simulator.py`

- expose all results generated by one environment transition in a form from which
  newly resolved block indices can be collected without duplication;
- keep transition behavior unchanged.

### `AllocRL/train.py`

- replace extractor choices and policy kwargs;
- filter active workspaces before environment creation;
- add seed and `gae_lambda` configuration;
- update architecture compatibility checks;
- use a new output directory for incompatible models.

### `AllocRL/alloc_env/callbacks.py`

- log resolved reward, terminal residual, terminal score, and CNN diagnostics;
- keep delay/dropout/success metrics.

### Notebook, export, evaluation, and visualization

- set Colab defaults to `candidate-cnn`, future horizon 4, and a new output path;
- update ONNX observation shapes and extractor compatibility;
- ensure evaluation and visualization use filtered workspace order and codes;
- optionally overlay candidate channel in debug visualization, not final placement
  playback.

### Tests

- replace attention-specific tests with ordered-future and candidate-CNN tests;
- retain unrelated simulator and environment regression tests.

## 11. Required tests

1. Candidate mask coordinates match the simulator placement.
2. A rotated candidate is rendered with swapped dimensions.
3. Observation generation does not mutate any block or workspace state.
4. An immediately unplaceable workspace has an all-zero candidate channel.
5. Swapping two valid future slots changes extractor output.
6. Changing padded future rows does not change extractor output.
7. Five configured active workspaces produce observation/action size five.
8. Episode reward sum equals terminal score.
9. Each resolved block contributes exactly once.
10. Identical seeds produce identical synthetic episodes and initial observations.
11. A PPO update changes candidate-CNN weights and produces a nonzero recorded
    gradient under a non-degenerate rollout.
12. Structured and fixed-grid extractors contain no convolution modules.
13. All extractors return finite features of the declared dimension.
14. MaskablePPO save/load and deterministic evaluation work for all extractors.
15. ONNX export works with the revised observation contract.

## 12. Migration and compatibility

The observation shape, workspace count, extractor classes, and reward timing are
incompatible with existing checkpoints. Auto-resume must reject old run configs
with a clear message. The local default remains `./output`, while the revised
Colab configuration uses the new explicit path
`/content/drive/MyDrive/CNN-RL-outputs/candidate_cnn_v1` so it cannot silently
resume the old attention model.

Historical logs remain readable when possible, but new reward and diagnostic
columns are versioned by the new output directory. No old model is silently loaded.

## 13. Implementation order

1. Add reward-conservation tests and resolved-result plumbing.
2. Add candidate-position records and four-channel observation tests.
3. Filter active workspaces and update action/observation contracts.
4. Implement the three extractors and remove attention code.
5. Update training, callbacks, notebook, export, evaluation, and visualization.
6. Run unit and regression suites.
7. Run short PPO smoke training for all three extractors.
8. Prepare the A-E ablation commands and fixed evaluation dataset.

Implementation is complete only when the required tests pass, short training can
save/load/evaluate each extractor, and reward conservation is demonstrated over
complete episodes.
