# Alignment with Group_20_slides_part2_1125.pptx

The supplied final deck has **23 slides**. The deck itself is not modified by this repository task.

| Slides | Subject | Repository counterpart |
|---|---|---|
| 1–3 | Task and motivation | README and teleoperation context |
| 4 | Observation, action and return | G1 configuration and environment |
| 5–7 | HIM/HIO | `rsl_rl/modules/him_estimator.py`, `him_actor_critic.py` |
| 8 | PPO | `rsl_rl/algorithms/him_ppo.py` |
| 9–10 | Three designs and pose curriculum | Method notes, sampler and curriculum variants |
| 11–12 | Height and knee rewards | Released absolute-error height kernel and knee shaping |
| 13 | Symmetry | Independent augmentation and loss controls |
| 14 | G1 network configuration | 76×6 history, 35-dimensional estimate, 12 outputs |
| 15 | Experiment settings | Reproducible configuration and evaluation scripts |
| 16–18 | Ablations | Corresponding variants; new measured curves require new runs |
| 19 | G1/GR-1 comparison | G1 only; no local GR-1 result |
| 20 | Operator interface | External-equipment context, no hardware implementation |
| 21 | Desktop task completion time | Outside the simulation-control experiment |
| 22 | Imitation learning | Outside this repository |
| 23 | Conclusions | Method summary and limitations |

## Wording checks for a simulation-only report

- HIM is a prior method. Slide 5 already names its source; it should not be described as unrelated to existing work.
- Slides 11–12 distinguish a paper expression from released code. This repository documents the actual code expression and its provenance.
- Slide 14 explicitly names the released implementation. The source relationship remains documented here.
- The values and curves on slides 16–19 are not backed by new Group 20 logs in this delivery. Treat them as referenced material unless independent logs are supplied.
- Slides 21–22 cannot substantiate a claim of simulation-only RL experiments. Hardware task times and imitation-learning success rates are separate experiments.

Suggested accurate scope statement:

> We study three training adaptations on a HIM+PPO humanoid-control implementation and provide a controlled simulation evaluation setup. Our repository focuses on G1 reinforcement learning; real-robot deployment and imitation learning are outside its scope.

Once training is actually completed, describe the runs using their recorded seeds, configuration, checkpoint hashes and measured results.
