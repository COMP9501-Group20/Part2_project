# COM9501_Group20_Project

**COMP9501 · Group 20 · Deep Reinforcement Learning for Robust Humanoid Control**

Chen Yuxuan · Deng Zihan · Han Yue · Jia Feiyu

[中文](README_zh.md) · [Simulation demonstrations](#simulation-demos) · [Installation](docs/SETUP.md)

<a id="overview"></a>

## Overview

This project studies humanoid lower-body control using a **Hybrid Internal Model (HIM)** and **Proximal Policy Optimization (PPO)**. The policy tracks locomotion and body-height commands while maintaining balance under independently changing upper-body postures.

The repository contains the G1 training task, Isaac Gym environment, learning code and experiment tools. It focuses on reinforcement learning and simulation; hardware construction, robot deployment and imitation learning are outside the implementation scope.

<a id="method"></a>

## Three training designs

1. **Upper-body pose curriculum:** gradually increase posture disturbance difficulty based on locomotion tracking performance.
2. **Height and knee shaping:** combine commanded-height tracking with knee-related rewards to support squatting and balance.
3. **Symmetry:** use mirrored experience and actor/critic consistency losses, with independent switches for controlled comparisons.

The HIM encoder estimates explicit velocity and a latent response from six observation frames. PPO learns 12 lower-body joint targets from these features. See [method details](docs/METHOD.md) and [implementation changes](docs/CHANGES.md).

<a id="simulation-demos"></a>

## Simulation demonstrations

Supplementary videos show locomotion, height adjustment and loco-manipulation in simulation. All five MP4 files are included in `media/videos/` and can be played locally after extraction.

| Demonstration | Simulator | Video |
|---|---|---|
| Unitree G1: locomotion and height control | Isaac Gym | [Play video](media/videos/g1.mp4) |
| Fourier GR-1: locomotion and height control | Isaac Gym | [Play video](media/videos/gr1.mp4) |
| Unitree G1: scene demonstration | GRUtopia | [Play video](media/videos/g1sim.mp4) |
| Fourier GR-1: scene demonstration | GRUtopia | [Play video](media/videos/gr1sim.mp4) |
| Loco-manipulation task demonstration | GRUtopia | [Play video](media/videos/g1tasksim.mp4) |

The runnable task in this repository is **G1 in Isaac Gym**. The GR-1 and GRUtopia videos demonstrate additional robot and scene settings; their training configurations and simulator integration are not included here.

<a id="quick-start"></a>

## Quick start

Target environment: Ubuntu 20.04/22.04, Python 3.8, Isaac Gym Preview 4 and an NVIDIA GPU. Follow [installation instructions](docs/SETUP.md), then run from the repository root:

```bash
# Inspect a training plan without Isaac Gym or a GPU.
python scripts/run.py train --variant full --seed 0 --dry-run

# Check simulator integration with a small run.
python scripts/run.py train --variant full --num-envs 32 --iterations 2 --headless

# Train the full method.
python scripts/run.py train --variant full --seed 0 --num-envs 4096 --iterations 5000 --headless

# Evaluate a checkpoint, matching its variant and training seed.
python scripts/run.py evaluate --variant full --seed 0 --checkpoint runs/YOUR_RUN/model_4999.pt --num-envs 1000 --horizon 20

# Watch the policy in the simulator.
python scripts/run.py play --variant full --checkpoint runs/YOUR_RUN/model_4999.pt
```

Replace `YOUR_RUN` with the actual training directory. `--output` selects a new run directory and refuses an existing one. Training saves TensorBoard events, checkpoints and `run.json`. Evaluation requires the checkpoint's matching training manifest and uses deterministic mean actions.

<a id="experiments"></a>

## Ablation experiments

| Variant | Pose curriculum | Height / knee weights | Mirrored data | Symmetry loss |
|---|---|---|---|---|
| `full` | Full-support | 2 / -0.75 | Yes | Yes |
| `no_curriculum` | Full difficulty from start | 2 / -0.75 | Yes | Yes |
| `range_curriculum` | Range-limited | 2 / -0.75 | Yes | Yes |
| `no_knee` | Full-support | 2 / 0 | Yes | Yes |
| `no_knee_high_height` | Full-support | 4 / 0 | Yes | Yes |
| `augmentation_only` | Full-support | 2 / -0.75 | Yes | No |
| `loss_only` | Full-support | 2 / -0.75 | No | Yes |
| `no_symmetry` | Full-support | 2 / -0.75 | No | No |
| `baseline` | Full difficulty from start | 0 / 0 | No | No |

`baseline` keeps the same G1 task and HIM+PPO architecture while disabling the three training additions. The higher height weight is set to 4. Default training seeds are `0, 1, 2`; evaluation uses 1000 environments, a 20-second horizon and the same full-difficulty upper-body sampler for every variant.

```bash
# Print all 27 jobs: 9 variants and 3 training seeds.
python scripts/sweep.py

# Run selected experiments sequentially.
python scripts/sweep.py --variants full no_curriculum no_knee --seeds 0 1 2 --execute

# Summarize measured evaluations and generate a figure.
python scripts/summarize.py runs/EVAL_0/metrics.json runs/EVAL_1/metrics.json runs/EVAL_2/metrics.json --output results/summary.csv --plot results/summary.png
```

The summary reports means and sample standard deviations across training seeds. See the [evaluation protocol](docs/EXPERIMENTS.md) for metric definitions and checkpoint selection.

<a id="structure"></a>

## Repository structure

| Path | Contents |
|---|---|
| `configs/experiments.json` | Experiment variants, seeds and evaluation settings |
| `project/` | Configuration and evaluation instrumentation |
| `scripts/` | Train, evaluate, play, sweep and summarize |
| `legged_gym/` | G1 Isaac Gym environment and robot assets |
| `rsl_rl/` | HIM, PPO, rollout storage and symmetry transforms |
| `tests/` | Algorithm and evaluation unit tests |
| `results/` | Experiment status and measured summaries |
| `media/` | Supplementary simulation video index |
| `docs/` | Installation, method and experiment documentation |
| `licenses/` | License notices and implementation history |

<a id="teleoperation"></a>

## Teleoperation context

The intended operator setup uses existing exoskeleton/glove and Apple Vision Pro devices. The simulation study represents operator inputs with sampled upper-body targets and locomotion/height commands. Device acquisition, image streaming and real-robot communication are not implemented here. See [interface scope](docs/TELEOPERATION.md).

<a id="validation"></a>

## Validation and run status

**10 CPU unit tests have passed.** They cover symmetry transforms, PPO updates, rollout returns, observation layout, evaluation statistics and checkpoint metadata.

```bash
python -m unittest discover -s tests -v
python scripts/run.py train --dry-run
```

This packaged version has not yet been retrained or validated end-to-end in Isaac Gym. No new checkpoints or quantitative evaluation files are included. The videos above are supplementary demonstrations. See [validation details](docs/VALIDATION.md) and [run status](results/status.json).

<a id="acknowledgements"></a>

## Acknowledgements and licenses

The learning stack uses [HIMLoco](https://github.com/Sage-PhysicalAI/HIMLoco), [RSL-RL](https://github.com/leggedrobotics/rsl_rl), [legged_gym](https://github.com/leggedrobotics/legged_gym), NVIDIA Isaac Gym and Unitree G1 robot assets. Component author notices and licenses are retained. See [acknowledgements](ATTRIBUTION.md) and [LICENSE](LICENSE).
