# Validation status

Assembled on 2026-11-26.

## Software checks

CPU tensor tests run on Windows with PyTorch 2.5.1. They cover:

- Mirroring twice recovers actor histories, critic observations and actions.
- Mirrored samples use the executed action's mirror.
- Original/mirrored GAE branches use separate final values.
- Single-branch terminal GAE stops bootstrapping.
- All four augmentation/regularization combinations complete a finite PPO update.
- The successor encoder receives the intended observation slice.
- Evaluation ignores states after the first failure/reset.
- Every variant uses identical full-difficulty evaluation settings.
- Summary aggregation calculates sample SD and rejects duplicate seeds/mismatched protocols.

The command is `python -m unittest discover -s tests -v`. These are synthetic software tests, not robot-performance experiments. Configuration dry-runs and sweep generation are also checked without Isaac Gym.

## Not run

- Isaac Gym physics integration, GPU rollout and full training.
- Checkpoint playback in the simulator.
- Real-robot deployment or teleoperation-device integration.

The available Windows environment has no Isaac Gym installation. Run the 32-environment integration smoke test in `SETUP.md` before a long experiment. The suggested Python 3.8 / legacy PyTorch GPU environment is separate from the CPU test environment used during assembly.
