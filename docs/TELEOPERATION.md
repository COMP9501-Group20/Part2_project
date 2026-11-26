# Teleoperation context and scope

The intended operator setup uses existing equipment: an exoskeleton/glove interface for upper-body targets, an Apple Vision Pro interface for viewing or operator interaction, and locomotion/height command input. The course repository focuses on learning the lower-body controller, not designing those devices.

## Logical inputs

| Input | Controller meaning | Current implementation |
|---|---|---|
| Upper-body joint targets | Independent posture changes that affect dynamics | Sampled and interpolated in Isaac Gym |
| Forward/lateral speed and yaw rate | Desired planar motion | Sampled command distribution |
| Body height | Desired stance/squat height | Sampled height commands |
| Vision Pro / image feedback | Operator situational awareness | Context only; no device/image driver |
| Exoskeleton/gloves | Source of upper-body targets in an integrated system | Context only; no acquisition driver |

Sampled commands let us study the RL problem without a connected operator device. They do not demonstrate that the external equipment has been integrated. This repository does not claim successful robot deployment, measured teleoperation completion times, or device-specific tracking performance.

If device integration is added later, document its source, joint ordering, units, joint limits, timestamps and evaluation separately. The current 27-joint simulation model and its 12-action policy should not be used as an undocumented real-robot interface.
