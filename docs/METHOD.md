# Learning method

## Observation and action

Each G1 observation has 76 features: four commands `(vx, vy, yaw rate, height)`, three IMU angular velocities, three projected-gravity values, 27 joint-position offsets, 27 joint velocities, and 12 previous lower-body actions. Six frames form a 456-dimensional actor history. The critic receives the current 76 features plus three simulator linear-velocity labels (79 total).

The actor outputs 12 lower-body joint offsets. Lower-body PD gains convert target positions into torques. The upper body's sampled targets are controlled separately in the simulator. The robot remains dynamically coupled, so whole-body feedback matters.

## HIM and HIO

The Hybrid Internal Model encodes history into a three-dimensional explicit velocity estimate and a normalized 32-dimensional latent response. The actor input is `76 + 3 + 32 = 111`. Actor and critic hidden widths are `512, 256, 256`.

Hybrid Internal Optimization trains the explicit branch against simulator velocity labels using squared error. The implicit branch compares history and successor embeddings with 64 learned prototypes. Balanced Sinkhorn assignments become fixed targets for the swapped prediction loss. The successor encoder is a training-only component.

This repository corrects the successor slice to `next_critic_obs[:, :76]`, consistent with this environment's layout `[76 observation features, 3 velocities]`. The previous slice `[:, 3:79]` mixed the final privileged velocity labels into the successor observation. See the regression test in `tests/test_core.py`.

## PPO

PPO optimizes the clipped policy-ratio objective and the critic objective, with entropy regularization. Defaults are clipping `0.2`, discount `0.99`, GAE `0.95`, initial learning rate `1e-3`, five epochs, four minibatches and 50 policy steps per environment per rollout. The policy rate is 50 Hz. HIM features are detached for PPO, so HIO and PPO have separate optimization paths.

## Three training choices

### 1. Upper-body curriculum

The implementation updates upper-body targets every second, interpolating between consecutive targets. A tracking-performance gate increases difficulty `rho` by `0.05`, capped at `1`.

The full-support sampler draws a truncated exponential scale and then a uniform joint ratio inside that scale. At low difficulty it favors small movements while retaining full support. The implemented formula uses `20*(1-0.99*rho)`, so `rho=1` is near-uniform for the outer scale, not the exact analytic uniform limit. The product of two sampled ratios is not uniform joint sampling.

`no_curriculum` fixes `rho=1` from the start. `range_curriculum` samples joint ratios uniformly in `[0,rho]`, a deliberately different support schedule. Every evaluation uses the same full-support sampler at `rho=1`.

### 2. Height and knee shaping

The code uses `exp(-4*abs(height_error))`, with a default scale of `2`. It measures pelvis-to-lowest-foot height plus the ankle sole offset. One third of sampled command modes involve squatting. The knee term normalizes knee position between configured target limits, couples deviation from the midpoint to root-height error, and has a negative reward scale of `-0.75`.

`no_knee` removes that term. `no_knee_high_height` also raises the height reward scale to `4`, a documented local comparison setting. `baseline` removes both height-related terms while retaining the same task/command observations.

### 3. Symmetry

Mirroring swaps left/right joint indices and applies the required signs to commands, angular velocity, gravity, joint state and actions. Every history frame is transformed consistently. Actor regularization compares the mirrored-input action mean to the mirrored original action mean; critic regularization encourages equal values.

Two independent switches control mirrored rollout data and regularization. Mirrored transitions use the mirror of the **executed** action, its log probability under the mirrored observation, mirrored successor observations, and branch-specific value bootstraps. Original and mirrored GAE streams stay separate. The estimator trains once on each stored sample.

All variants use the same corrected algorithm path. Compare results from runs using the same implementation and evaluation settings.
