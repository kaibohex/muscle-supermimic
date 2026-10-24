# muscle-supermimic

[SuperMimic](https://arxiv.org/abs/2610.28480) (one differentiable world model, BPTT through
imagined rollouts) driving whole-body musculoskeletal models -- **MS-Human-700** (700 Hill-type
muscles, 37 independent joint DoF + free root) and MuscleMimic's **MyoFullBody** (354 muscles, 31
independent joint DoF + free root, fingers disabled) -- built on [mjlab](https://github.com/mujocolab/mjlab)
(MuJoCo-Warp). The algorithm follows the unofficial reproduction
[supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) (MIT). The task (MDP) follows
**MuscleMimic** as supermusclemimic runs SuperMimic on it -- the setup in which SuperMimic trains
stably on a muscle model: MuscleMimic's 100 Hz control, observation, goal, MimicReward and deviation
terminations (see [MDP](#mdp-musclemimics)).

Status: MS-Human-700 runs on a single A100 40 GB with 4096 envs; MyoFullBody is tested on CPU only so
far. Motion capture data for MS-Human-700 is still to come; a msgym walking clip is used for testing.
MyoFullBody trains on MuscleMimic's retargeted AMASS clips.

## Layout (mjlab conventions)

| Path | What |
|---|---|
| `third_party/MS-Human-700/` | git submodule, the upstream model ([LNSGroup/MS-Human-700](https://github.com/LNSGroup/MS-Human-700)); `MS_HUMAN_700_ROOT` overrides the location |
| `src/muscle_supermimic/assets/robots/model_info.py` | joint/muscle bookkeeping shared by the robots (`ModelInfo`: root, independent and coupled joints, muscles; `root_offset`) |
| `src/muscle_supermimic/assets/robots/mshuman700/` | `EntityCfg`: the 6 pelvis DoF become a free joint, scene decoration stripped, muscles wrapped by `XmlActuatorCfg` |
| `src/muscle_supermimic/assets/robots/myofullbody/` | `EntityCfg` for MyoFullBody, loaded from the `musclemimic-models` package as MuscleMimic does: scene stripped, default pose turned to face +x, fingers disabled as in MuscleMimic |
| `src/muscle_supermimic/tasks/tracking/tracking_env_cfg.py` | the task's `ManagerBasedRlEnvCfg`: observations, actions, command, rewards, terminations, sim |
| `src/muscle_supermimic/tasks/tracking/mdp/` | `kinematics.py` differentiable FK, `muscle.py` activation dynamics, `muscle_geometry.py` muscle length / velocity / force, `motion.py` reference library, `heads.py` analytic MDP, `commands.py` motion command (reset sampling), `actions.py`, `observations.py`, `rewards.py`, `terminations.py` |
| `src/muscle_supermimic/tasks/tracking/config/mshuman700/` | robot binding, registration of `Mjlab-SuperMimic-Flat-MS-Human-700` and `...-Locked-Hands` |
| `src/muscle_supermimic/tasks/tracking/config/myofullbody/` | robot binding, registration of `Mjlab-SuperMimic-Flat-MyoFullBody` |
| `src/muscle_supermimic/tasks/tracking/scripts/` | `convert_motion.py` (`ms-convert-motion`), `evaluate.py` (`ms-eval`) |
| `src/muscle_supermimic/rl/supermimic/` | the SuperMimic runner (`runner.py`), world model / policy / critic (`networks.py`), replay buffer, config |
| `tests/` | FK, activation dynamics and muscle length / velocity / force vs MuJoCo (both robots); muscle-joint map vs finite-difference moment arms; MyoFullBody vs MuscleMimic's finger removal and its retargeted body poses; simulator vs analytic heads; a tiny end-to-end training run (all tasks) |

The tasks are registered through the `mjlab.tasks` entry point, so mjlab's own `train` / `play` CLIs
run them with the SuperMimic runner (`register_mjlab_task(..., runner_cls=SuperMimicRunner)`).

How the two sides share one MDP: every observation / reward / termination term of the env is a thin
manager wrapper around an analytic function of a state tuple `(p, R, v, w, q, qd, act)`, attached as
`term.func.analytic`. The simulator labels transitions through the managers; the runner evaluates
the same functions, with the env cfg's weights, stds and thresholds, on imagined states. The tests
check that both give the same rewards, observation history and goal.

## Setup

```bash
git clone --recursive <this repo>          # or: git submodule update --init --recursive
uv sync                                    # mjlab 1.6.0 (+ torch, mujoco-warp), musclemimic-models 1.0.6
uv run pytest                              # CPU, ~2 min (test_env.py needs the test clips, below)
```

## Motion data

A clip is one `.npz`; a directory of them (searched recursively, clips named by their relative path)
is a motion library. Three formats are read:

* **canonical** (what `ms-convert-motion` writes): `fps`, `root_pos` (T,3) pelvis origin in the world
  [m], `root_quat` (T,4) wxyz of the *canonical root frame* (x forward, y left, z up; identity =
  standing upright facing +x; it is the pelvis body frame times `ROOT_OFFSET`), `joint_pos` (T,J) [rad]
  and `joint_names` (J,). Columns are matched by name, so joints the model lacks (the wrists of the
  locked-hands task) are simply dropped; without `joint_names`, 37 columns in this order
  (`ModelInfo.indep_names` of the full model):

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

Clips are resampled to the control rate (100 Hz); velocities come from finite differences, as in
supermimic_vibed.

Test clip (msgym's one walking cycle, tiled into 8 cycles / 9.4 s):

```bash
mkdir -p data/msgym
curl -L -o data/msgym/walking_gait.npz https://github.com/LNSGroup/msgym/raw/main/msgym/motion_data/walking_gait.npz
uv run ms-convert-motion data/msgym/walking_gait.npz data/motions/walk.npz --cycles 8
```

MyoFullBody clips: MuscleMimic's GMR-retargeted AMASS motions
([amathislab/musclemimic-retargeted](https://huggingface.co/datasets/amathislab/musclemimic-retargeted),
KIT: 1077 clips, 1.7 h at 100 Hz; gated, accept the terms on the page first), read straight from the download:

```bash
uvx --from huggingface_hub hf auth login
uvx --from huggingface_hub hf download amathislab/musclemimic-retargeted --repo-type dataset --include "MyoFullBody/gmr/KIT/*" --local-dir data/musclemimic
mkdir -p data/myofullbody && cp data/musclemimic/MyoFullBody/gmr/KIT/314/walking_medium08_poses.npz data/myofullbody/walk.npz  # test clip
```

These caches were retargeted with musclemimic-models <= 1.0.5, whose left-knee coupler
`knee_angle_rotation3_l` had the opposite sign; 1.0.6 (pinned here, and what MuscleMimic and
supermusclemimic lock) mirrors the right knee. The clips' stored `knee_angle_rotation3_l` is therefore
off by up to 0.48 rad, which the motion library ignores like every coupled joint: the reference left
shank and foot follow the installed model's coupler, up to 4 cm away from the poses stored with the
test clip (`tests/test_myofullbody.py`). Everything else matches them to < 1 mm.

## Tasks

| Task | Joint DoF | Muscles |
|---|---|---|
| `Mjlab-SuperMimic-Flat-MS-Human-700` | 37 + free root | 700 |
| `Mjlab-SuperMimic-Flat-MS-Human-700-Locked-Hands` | 33 + free root | 694 |
| `Mjlab-SuperMimic-Flat-MyoFullBody` | 31 + free root | 354 |

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

**MyoFullBody** (`get_myofullbody_cfg()`), as MuscleMimic configures it for full-body tracking: the model
comes from `musclemimic_models.get_xml_path("myofullbody")`; MuscleMimic's `disable_fingers` deletes
the 40 finger joints and the 62 finger muscles with their tendons (wrists stay); the 75 explicit
self-contact pairs (arms vs thorax and legs, leg vs leg) stay, robot geoms otherwise touch the ground
only; two abdominal slides held at zero by `<equality>` are coupled joints with a constant. Upstream's
default pose faces -y; the entity turns it to face +x. The root is the free joint's body `Full Body`, to
which the pelvis is welded (same origin). Foot touch is read like MS-Human-700's, as the ground's net
force on the bodies of MyoFullBody's four touch sensor sites (calcn and toes per side). MuJoCo-Warp budget
per env as in MuscleMimic: `nconmax` 96, `njmax` 768.

## Usage

```bash
# train (Table A.5 defaults: 4096 envs, buffer 1000 steps/env, UTD 0.4, H=32, ...)
uv run train Mjlab-SuperMimic-Flat-MS-Human-700 --env.commands.motion.motion-file data/motions
# fewer envs / smaller buffer and imagination batch for a smaller GPU
uv run train Mjlab-SuperMimic-Flat-MS-Human-700 --env.commands.motion.motion-file data/motions \
    --env.scene.num-envs 1024 --agent.buffer-steps 500 --agent.pi-batch 2048
# evaluate checkpoints (App. F.1: every clip once from its first frame; normalized return, failed clips)
uv run ms-eval --run-dir logs/rsl_rl/mshuman700_tracking/<run> --motion-file data/motions --watch
uv run ms-eval --checkpoint logs/rsl_rl/mshuman700_tracking/<run>/model_014000.pt --motion-file data/motions --record traj.npz
# watch a policy in mjlab's viewer
uv run play Mjlab-SuperMimic-Flat-MS-Human-700 --checkpoint-file <model.pt> --motion-file data/motions/walk.npz
# MyoFullBody on MuscleMimic's KIT clips (logs to logs/rsl_rl/myofullbody_tracking/)
uv run train Mjlab-SuperMimic-Flat-MyoFullBody --env.commands.motion.motion-file data/musclemimic/MyoFullBody/gmr/KIT
uv run ms-eval --task Mjlab-SuperMimic-Flat-MyoFullBody --run-dir logs/rsl_rl/myofullbody_tracking/<run> --motion-file data/myofullbody
```

Runs log to `logs/rsl_rl/mshuman700_tracking/<date>/` (`train_log.csv`, tensorboard, `model_*.pt` every
2000 iterations). mjlab's CLI writes tuples in Python syntax: `--agent.wm-hidden "(1024,1024,1024)"`.

Memory with the defaults, roughly: replay buffer ~14 GB (4096 x 1000 transitions, 700-D activations
and excitations in fp16), imagination BPTT (4096 x 32 steps) a few tens of GB. `--agent.buffer-steps`,
`--agent.pi-batch` and `--env.scene.num-envs` are the knobs.

## MDP (MuscleMimic's)

Every term is an analytic function of the state, evaluated identically on simulator and imagined
states. MuscleMimic's 17 mimic sites are the frames of 17 tracked bodies, pelvis first (the main site):
on MyoFullBody the sites' own bodies (pelvis, lumbar1, head, humerus, ulna, lunate, femur, tibia, talus,
toes), on MS-Human-700 bodies whose origins sit at the joints the sites mark (pelvis, femur, tibia, calcn,
toes x2, thoracic12, head_neck, humerus, ulna, hand x2).

* **Control**: 100 Hz (2 ms x 5 substeps), as MyoFullBody.
* **Observation** (decoder, no history): free-joint height and quaternion, every joint position, the
  free joint's qvel, every joint velocity, muscle length / velocity / force / excitation / activation,
  four foot touch values (ground force on heel and toes), and the root-relative site positions, rotation
  vectors and 6D velocities (GoalTrajMimic's state-dependent part).
* **Goal** (encoder): GoalTrajMimic's concise lookahead -- the reference's root-relative site positions
  at the current frame and 4 frames 0.2 s apart, with root position and qvel deltas -- and the motion phase.
* **Reward**: MimicReward with the full-body weights (qpos 0.1, qvel 0.1, root pos 0.1, root vel 0.1,
  site rpos 0.6, site rquat 0.01, site rvel 0.1 + 0.1; kernels 10 / 2 / 10 / 10 / 100 / 10 / 0.1), times
  supermusclemimic's reward scale 0.1; no penalties (all off in the full-body configuration).
* **Termination**: mean root-relative site deviation > 1 m, root position deviation > 1 m, root
  orientation deviation > 1 rad (sigmoid-softened with 5 % widths in imagination); time-outs at 10 s
  and at the clip end, bootstrapped.

Muscle length, velocity and force on imagined states come from a torch replica of MuJoCo's spatial
tendons (site-site segments, sphere / cylinder wraps with side sites and inside wraps) and Hill muscle
curves (`mdp/muscle_geometry.py`, equal to MuJoCo's `actuator_length / velocity / force` to 1e-9 /
1e-8 / 1e-6). Foot touch, which depends on contacts, is predicted by an extra world-model head.

## Algorithm (supermimic_vibed, as supermusclemimic adapts it to muscles)

* **Reset sampling**: 1 s bins, 50% uniform / 40% softmax(-25 G_b / G_max) / 10% its running average.
* **World model** MLP(2048,2048,2048) + LayerNorm + SiLU on heading-invariant features and the
  activations before and after the step, frame-local increments plus a foot-touch head, k=8-step
  autoregressive loss with per-component error floors (+ touch MSE). Activations are integrated
  analytically with MuJoCo's own substeps (they match the simulator to float precision).
* **Policy** encoder MLP(2048,1024,512,256) → z∈R^32 (tanh) + decoder MLP(2048,1024,512,256) →
  excitations e = sigmoid(·) in [0, 1], initialized at 0.1; exploration noise N(0, 0.1²) on e, clipped.
* **Critic** MLP(256,128) on the decoder observation, the goal and privileged reference errors (root
  position in the heading frame, heading error, joint position and velocity errors), GAE(λ=0.95)
  targets, 5 epochs x 4 minibatches. BPTT over H=32 imagined steps (0.32 s) with soft survival and
  critic bootstrap, each step recomputed in the backward pass; Adam 2e-4 (policy β=(0.9,0.95)), cosine
  decay to 0.03x over 250k iterations, grad clip 1.0 (+0.1 element-wise for the policy), γ=0.99, 500
  world-model-only iterations, bf16 MLPs, 4096 envs, imagination batch 4096.
* **Evaluation**: App. F.1 as in supermimic_vibed.

Model-specific: coupled joints come from the `<equality>` polynomials (the torch FK replicates
`mj_kinematics` to 1e-9). MS-Human-700: 37 independent and 42 coupled joints; one free joint for the root
instead of 6 OpenSim DoF; robot-robot contacts disabled; upstream, the arms have no collision geoms.
MyoFullBody: 31 independent and 51 coupled joints; MuscleMimic's self-contact pairs kept. Optional, off by default
(`--agent.critic-target-alpha`, `--agent.value-clip`): a SHAC-style target critic and clamping
bootstrapped values to the attainable maximum.

Why MuscleMimic's MDP: with supermimic's own (BeyondMimic reward with action-rate and joint-limit
penalties, height/tilt terminations, 50 Hz, 10-frame proprioceptive history) every run lost its critic
200-400 iterations after policy updates started. The penalties, summed over 694 muscles (G1: 29
actuators) and over the joints a collapsing body pins at their limits, were as large as the tracking
reward, so staying alive was worth nothing (imagined survival never rose).

By request, for now:

* **No randomization of any kind**: no friction / encoder offset / CoM domain randomization (the sim
  context ξ is removed from the world model and critic), no observation noise, no reset noise. Exploration
  noise on the actions and the sampling of reset frames are part of the algorithm and stay.

## Attribution

* Method: Q. Liao et al., *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World
  Model*, CoRL 2026, arXiv:2610.28480. Implementation ported from supermimic_vibed by Chong Zhang (MIT,
  Copyright (c) 2026 ETH Zurich, Chong Zhang); ported files say so in their docstrings.
* MS-Human-700 (Apache-2.0, LNSGroup), msgym walking data (Apache-2.0), mjlab (Apache-2.0).
* MuscleMimic (Apache-2.0, [amathislab/musclemimic](https://github.com/amathislab/musclemimic); license
  in `third_party/musclemimic/LICENSE`): the MyoFullBody entity follows its environment
  (`musclemimic/environments/humanoids/myofullbody.py`): the finger joint / muscle lists and their
  removal, the mimic-site bodies tracked. Li et al., *Towards Embodied AI with MuscleMimic*, arXiv:2603.25544.
* MyoFullBody model: the `musclemimic-models` package (Apache-2.0), installed, not vendored. Its NOTICE
  lists what it derives from: MyoSuite / myo_sim (Apache-2.0) and OpenSim models, among them MoBL-ARMS
  (arm and hand), released for **non-commercial use only**.
* MyoFullBody motions: MuscleMimic's retargeted AMASS (KIT) data, under AMASS's own license; not vendored.
