# Implementation changes

## Repository organization

- Keep the G1 Isaac Gym task, robot-description assets and HIM/PPO learning code.
- Omit hardware design, robot deployment, deployment export scripts and promotional images.
- Add `project/`, `scripts/`, the variant matrix, tests, and course-specific documentation.
- Retain component author notices, license texts and per-file implementation history.

## Algorithm and experiment changes

1. **Independent symmetry controls.** `use_flip` selects mirrored rollout data. `use_symmetry_loss` independently selects actor/critic consistency losses, enabling all four combinations.
2. **Executed-action mirroring.** A synthetic mirrored transition uses the mirror of the actual action instead of a separately sampled action unrelated to the observed transition. Old log probability is computed for that mirrored action under the mirrored observation.
3. **Rollout branch handling.** Storage supports one or two streams. GAE uses a distinct final value for each stream. Disabling augmentation no longer leaves an implicitly doubled buffer. Incomplete rollouts are rejected.
4. **Estimator sample handling.** Mirrored samples already in storage are not concatenated a second time for estimator training. The successor observation slice is corrected to the G1 critic layout, excluding privileged velocity labels.
5. **Range-limited curriculum alternative.** Add a sampler option for the comparison in slide 16. Default behavior retains the full-support sampler. Evaluation overrides all variants to identical full difficulty.
6. **Evaluation instrumentation.** Capture errors before automatic reset and stop each measured trajectory at its first physical failure. Record checkpoint hashes and protocol metadata. Aggregate across training seeds with sample SD.
7. **Usability fixes.** Use local TensorBoard by default, explicit checkpoint paths, correct periodic checkpoint iteration metadata, remove a debugger stop in code-state logging, and raise a descriptive error for NaN rewards.

The mirror, storage and estimator changes can affect learning behavior. Retrain every ablation variant using this same implementation before comparing quantitative results.

## Deliberate limitations

- Only the provided G1 task is registered. There is no tested GR-1 configuration.
- No real-robot or external-device communication has been included or tested.
- No imitation-learning model or autonomous manipulation dataset is included.
- Complete Isaac Gym integration remains to be tested on the documented Linux environment.
