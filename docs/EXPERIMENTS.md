# Simulation experiment protocol

## Questions

1. Does a performance-gated upper-body pose curriculum improve tracking at full disturbance difficulty?
2. Does knee shaping help height control without the same velocity tradeoff as simply increasing the height reward?
3. Do data augmentation and symmetry regularization improve different aspects of learning?

The variant matrix is in `configs/experiments.json`. Train each variant independently with seeds `0, 1, 2`. The default 5000 iterations are an initial experiment budget, not a claim that the method has converged. Keep physical environment steps, network dimensions, optimizer settings and domain randomization equal across variants. Augmentation increases the number of stored samples and computation; record runtime when comparing efficiency.

## Training artifacts

Each run creates a new directory containing:

- `run.json`: mode, variant, seed, full environment/algorithm configuration, package version, code revision and completion status.
- `events.out.tfevents.*`: local TensorBoard training curves.
- `model_*.pt`: actor, critic, HIM and optimizer state.

Use `tensorboard --logdir runs` to view training. A two-iteration smoke run tests integration only; it is not a trained controller. The sweep script does not automatically launch 27 runs unless `--execute` is supplied.

## Evaluation

For each trained seed, evaluate a specified checkpoint using 1000 environments, a 20-second horizon, evaluation seed `12345`, full-support upper targets at difficulty `1`, and the same domain randomization/noise settings. Evaluations use the actor mean. Pair the selected checkpoint iteration across variants; do not select each method's best checkpoint using test results.

The wrapper samples metrics **before automatic reset**. Once an environment fails, subsequent reset episodes are excluded. The first failure step is included. A single synchronized episode is counted for each environment, capped at the horizon. Tracking errors are time-averaged within each episode and then averaged equally over environments. Always report survival alongside errors, because early failures shorten the observed trajectory.

| Metric key | Definition |
|---|---|
| `linear_velocity_error_m_s` | Euclidean error in body-frame x/y velocity |
| `yaw_velocity_error_rad_s` | Absolute yaw-rate tracking error |
| `height_error_m` | Absolute commanded-height error, with the same foot/sole convention as the height reward |
| `policy_symmetry_mse` | Sum of squared action-mean differences between mirrored input and mirrored original output |
| `living_time_s` | Time to first physical termination, capped at the evaluation horizon |
| `survival_rate` | Fraction without physical termination during the evaluated horizon |

The velocity/height command used to measure a step is captured before that step's command resampling. This avoids scoring against a newly changed command that the action did not see.

These definitions specify the metrics produced by the evaluation scripts. Use the generated `metrics.json` files when comparing new runs.

## Summarizing and plotting

Evaluate each seed separately. Then pass those `metrics.json` files to `scripts/summarize.py`. The output contains the mean and **sample standard deviation across independent training seeds**. A single seed gets an empty standard deviation, not a false zero. The script rejects duplicate `(variant, training_seed)` records and mismatched evaluation settings.

To produce learning curves, evaluate the same selected checkpoints for every seed and variant. Summarize each checkpoint separately. Checkpoint identity and SHA-256 are recorded in each evaluation file. Default summary bar charts show the final selected checkpoint.

## Actual result status

No training or evaluation runs were supplied. `results/status.json` therefore starts at `not_run`. The assembly-time unit tests use synthetic tensors and are recorded under `docs/VALIDATION.md`; they are software checks, not simulation experiments.
