# Motion data

A clip is one `.npz`; a directory of them (searched recursively) or a `.txt` list of them (one path
per line, relative to the list's directory; `#` comments) is a motion library, its clips named by their
path relative to the directory they share. Clips are resampled to the control rate (100 Hz); velocities
come from finite differences, as in supermimic_vibed.

## Formats

* **canonical** (what `ms-convert-motion` and `ms-retarget-myofullbody` write): `fps`, `root_pos` (T,3)
  pelvis origin in the world [m], `root_quat` (T,4) wxyz of the *canonical root frame* (x forward, y left,
  z up; identity = standing upright facing +x; it is the pelvis body frame times `ROOT_OFFSET`),
  `joint_pos` (T,J) [rad] and `joint_names` (J,). Columns are matched by name, so joints the model lacks
  (the wrists of the locked-hands task) are simply dropped; without `joint_names`, 37 columns in this
  order (`ModelInfo.indep_names` of the full MS-Human-700):

  ```
  hip_flexion_r hip_adduction_r hip_rotation_r knee_angle_r ankle_angle_r subtalar_angle_r mtp_angle_r
  hip_flexion_l hip_adduction_l hip_rotation_l knee_angle_l ankle_angle_l subtalar_angle_l mtp_angle_l
  L5_S1_FE L5_S1_LB L5_S1_AR T12_L1_FE T12_L1_LB T12_L1_AR T1_head_neck_FE T1_head_neck_LB T1_head_neck_AR
  elv_angle_r shoulder_elv_r shoulder_rot_r elbow_flexion_r pro_sup_r deviation_r flexion_r
  elv_angle_l shoulder_elv_l shoulder_rot_l elbow_flexion_l pro_sup_l deviation_l flexion_l
  ```

  The other 42 joints (knee translations, patella, shoulder girdle, wrist) follow from these through
  the model's `<equality>` polynomials and are not part of the data.
* **MS-Human-700 native**: `qpos_traj` (T,85) in the upstream model's qpos layout (6 OpenSim pelvis
  DoF first) and `framerate`, e.g. the output of an IK/retargeting pipeline run on the upstream XML,
  or msgym's clips. The root pose is recovered with MuJoCo FK on the upstream model.
* **MuscleMimic trajectory**: `qpos` (T,nq) in the layout of `joint_names` (+ `jnt_type`), the root free
  joint first, and `frequency` -- MuscleMimic's retargeted MyoFullBody caches as they are. Coupled
  joints are ignored here too.

## Test clip

msgym's one walking cycle, tiled into 8 cycles / 9.4 s (MS-Human-700):

```bash
mkdir -p data/msgym
curl -L -o data/msgym/walking_gait.npz https://github.com/LNSGroup/msgym/raw/main/msgym/motion_data/walking_gait.npz
uv run ms-convert-motion data/msgym/walking_gait.npz data/motions/walk.npz --cycles 8
```

## MyoFullBody: MuscleMimic's retargeted AMASS (KIT)

MuscleMimic's GMR-retargeted AMASS motions
([amathislab/musclemimic-retargeted](https://huggingface.co/datasets/amathislab/musclemimic-retargeted),
KIT: 1080 clips, 1.7 h at 100 Hz; gated, accept the terms on the page first), read straight from the
download. MuscleMimic's split ships with the repo: `data/musclemimic/kit_kinesis_train.txt` (972 clips, its
`KIT_KINESIS_TRAINING_MOTIONS`, 1.5 h) to train on and `kit_kinesis_test.txt` (108 clips,
`KIT_KINESIS_TESTING_MOTIONS`) to validate on, listing the download's paths:

```bash
uvx --from huggingface_hub hf auth login
uvx --from huggingface_hub hf download amathislab/musclemimic-retargeted --repo-type dataset --include "MyoFullBody/gmr/KIT/*" --local-dir data/musclemimic
mkdir -p data/myofullbody && cp data/musclemimic/MyoFullBody/gmr/KIT/314/walking_medium08_poses.npz data/myofullbody/walk.npz  # test clip
```

These caches were retargeted with musclemimic-models <= 1.0.5, whose left-knee coupler
`knee_angle_rotation3_l` had the opposite sign; 1.0.6 (pinned here, as MuscleMimic pins it) mirrors the
right knee. The clips' stored `knee_angle_rotation3_l` is therefore off by up to 0.48 rad, which the motion
library ignores like every coupled joint: the reference left shank and foot follow the installed model's
coupler, up to 4 cm away from the poses stored with the test clip (`tests/test_myofullbody.py`). Everything
else matches them to < 1 mm.

## MS-Human-700: retargeted from MyoFullBody

`ms-retarget-myofullbody` retargets MyoFullBody clips onto MS-Human-700 (CPU; the 1080 KIT clips take
20 min with 6 processes):

```bash
uv run ms-retarget-myofullbody data/mshuman700_kit data/musclemimic/kit_kinesis_train.txt data/musclemimic/kit_kinesis_test.txt --workers 6
```

It writes canonical clips under `data/mshuman700_kit/` (paths relative to the KIT folder) and the two lists
next to them. Both models come from the same OpenSim models: the legs (hip, knee, ankle, subtalar, mtp) and
the shoulder, elbow and forearm have the same joints about the same axes, so those 27 angles carry over
(their segments turn exactly alike; positions differ by the knee's translation curves, < 2 cm, and the
shoulders' place on the thorax, ~4 cm), and so does the root pose (the pelvis frames coincide). The rest is
fitted, all frames of a clip at once (L-BFGS through the torch FK, so the couplers hold exactly): L5_S1 and
T12_L1 to the thorax's rotation relative to the pelvis (MyoFullBody bends five lumbar levels and keeps thorax
and head rigid; residual <= 0.01 deg in all but one clip, whose 40 deg of axial twist exceed MS-Human-700's
+-0.6 rad, 4 deg there), wrist flexion to the hand's rotation relative to the radius. The neck and wrist
deviation stay at 0: MS-Human-700's deviation axis is oblique and turns the hand the other way, and in the GMR
clips MyoFullBody's deviation mostly sits at its limit, an IK artifact. The hand residuals are that
deviation: a clip's worst frame is off by 10 deg (median over clips), at most 29 deg. Angles beyond
MS-Human-700's ranges (mostly ankle dorsiflexion past 0.52 rad and subtalar past 0.34 rad) are clipped: 1 %
of the values in the median clip, at most 4 %. Finally each clip is lowered by one height, 0.74 cm (median;
up to 0.93 cm), so that the feet touch the ground as in the source (MS-Human-700's foot capsules sit higher
than MyoFullBody's foot geoms on the same bones).

The retargeted clips derive from AMASS and fall under its license: keep them to yourself, do not
redistribute them.
