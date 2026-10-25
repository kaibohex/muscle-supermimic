# Models

| Task | Joint DoF | Muscles |
|---|---|---|
| `Mjlab-SuperMimic-Flat-MS-Human-700` | 37 + free root | 700 |
| `Mjlab-SuperMimic-Flat-MS-Human-700-Locked-Hands` | 33 + free root | 694 |
| `Mjlab-SuperMimic-Flat-MyoFullBody` | 31 + free root | 354 |

Both robots are trees of hinge and slide joints under one free joint, part of them tied to another joint by
a joint `<equality>` polynomial (the coupled joints, which the torch FK reproduces from `mj_kinematics` to
1e-9), driven by Hill-type muscles that each pull one spatial tendon. `ModelInfo`
(`assets/robots/model_info.py`) derives the root, the independent and coupled joints and the muscles from
the compiled model.

## MS-Human-700

The upstream model ([LNSGroup/MS-Human-700](https://github.com/LNSGroup/MS-Human-700), a git submodule in
`third_party/MS-Human-700/`; `MS_HUMAN_700_ROOT` overrides the location) is loaded through `MjSpec` and
changed in these ways: the six OpenSim pelvis DoF become one free joint (no gimbal lock; mjlab needs a free
joint for a floating base); the standalone scene (floor, lights, cameras, keyframe) goes; robot-robot
contacts are disabled, robot-ground contacts stay (upstream, the arms have no collision geoms). Muscles,
wrapping tendons, couplers, joint damping and stiffness are upstream. 37 independent and 42 coupled joints.

**Locked hands** (`get_mshuman700_cfg(lock_hands=True)`): every joint in the `proximal_row_{r,l}`
subtrees is deleted -- per side the wrist's `deviation` / `flexion` and the two `wrist_hand` joints
coupled to them -- so each hand is welded to its radius in the neutral pose (q = 0, which satisfies the
couplers); the 4 couplers go with them. A muscle is deleted when its tendon path crosses no joint other
than these (`muscle_spans`: path points / wrap surfaces on both sides of a joint; the tests check it
against finite-difference moment arms for all 700 muscles). That removes APL, EPB and FPL on both sides.
The other 42 muscles crossing the wrist also cross the elbow and/or the radioulnar joint (`pro_sup`)
and stay; 16 of them (per side FDP x4, FDS index/middle, EIP, EPL) reach `pro_sup` only, with moment
arms of 0.3-3.5 mm there against 7-42 mm at the wrist. Reference clips keep their wrist columns; they are
ignored. Motion command, FK, observations and action size follow the entity's variant automatically.

## MyoFullBody

`get_myofullbody_cfg()` configures MyoFullBody as MuscleMimic does for full-body tracking: the model comes
from `musclemimic_models.get_xml_path("myofullbody")` (the `musclemimic-models` package, installed, not
vendored); MuscleMimic's `disable_fingers` deletes the 40 finger joints and the 62 finger muscles with their
tendons (wrists stay); the 75 explicit self-contact pairs (arms vs thorax and legs, leg vs leg) stay, robot
geoms otherwise touch the ground only; two abdominal slides held at zero by `<equality>` are coupled joints
with a constant. 31 independent and 51 coupled joints. Upstream's default pose faces -y; the entity turns it
to face +x. The root is the free joint's body `Full Body`, to which the pelvis is welded (same origin). Foot
touch is read like MS-Human-700's, as the ground's net force on the bodies of MyoFullBody's four touch sensor
sites (calcn and toes per side). MuJoCo-Warp budget per env as in MuscleMimic: `nconmax` 96, `njmax` 768.
