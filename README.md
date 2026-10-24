# muscle-supermimic

[SuperMimic](https://arxiv.org/abs/2610.28480) (one differentiable world model, BPTT through
imagined rollouts) driving the **MS-Human-700** whole-body musculoskeletal model (700 Hill-type
muscles, 37 independent joint DoF + free root), built on [mjlab](https://github.com/mujocolab/mjlab)
(MuJoCo-Warp). The algorithm follows the unofficial reproduction
[supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) (MIT); its MDP (observations,
rewards, terminations, reset sampling) and hyperparameters are kept as they are, and only what the
muscle-actuated plant requires is changed (see [Deviations](#deviations-from-supermimic_vibed)).

Status: the algorithm runs end to end (CPU tests below). It has **not been trained on a GPU yet**:
throughput, memory and learning results are unverified. Motion capture data is still to come; a
msgym walking clip is used for testing.

## Layout (mjlab conventions)

| Path | What |
|---|---|
| `third_party/MS-Human-700/` | git submodule, the upstream model ([LNSGroup/MS-Human-700](https://github.com/LNSGroup/MS-Human-700)); `MS_HUMAN_700_ROOT` overrides the location |
| `src/muscle_supermimic/assets/robots/mshuman700/` | `EntityCfg`: the 6 pelvis DoF become a free joint, scene decoration stripped, muscles wrapped by `XmlActuatorCfg`; joint/muscle bookkeeping (`ModelInfo`) |
| `src/muscle_supermimic/tasks/tracking/tracking_env_cfg.py` | the task's `ManagerBasedRlEnvCfg`: observations, actions, command, rewards, terminations, sim |
| `src/muscle_supermimic/tasks/tracking/mdp/` | `kinematics.py` differentiable FK, `muscle.py` activation dynamics, `motion.py` reference library, `heads.py` analytic MDP, `commands.py` motion command (reset sampling), `actions.py`, `observations.py`, `rewards.py`, `terminations.py` |
| `src/muscle_supermimic/tasks/tracking/config/mshuman700/` | robot binding, registration of `Mjlab-SuperMimic-Flat-MS-Human-700` and `...-Locked-Hands` |
| `src/muscle_supermimic/tasks/tracking/scripts/` | `convert_motion.py` (`ms-convert-motion`), `evaluate.py` (`ms-eval`) |
| `src/muscle_supermimic/rl/supermimic/` | the SuperMimic runner (`runner.py`), world model / policy / critic (`networks.py`), replay buffer, config |
| `tests/` | FK and activation dynamics vs MuJoCo; muscle-joint map vs finite-difference moment arms; simulator vs analytic heads; a tiny end-to-end training run (both tasks) |

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
uv sync                                    # mjlab 1.6.0 (+ torch, mujoco-warp)
uv run pytest                              # CPU, ~1 min (test_env.py needs data/motions/walk.npz, below)
```

## Motion data

A clip is one `.npz`; a directory of them is a motion library. Two formats are read:

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

Clips are resampled to 50 Hz; velocities come from finite differences, as in supermimic_vibed.

Test clip (msgym's one walking cycle, tiled into 8 cycles / 9.4 s):

```bash
mkdir -p data/msgym
curl -L -o data/msgym/walking_gait.npz https://github.com/LNSGroup/msgym/raw/main/msgym/motion_data/walking_gait.npz
uv run ms-convert-motion data/msgym/walking_gait.npz data/motions/walk.npz --cycles 8
```

## Tasks

| Task | Joint DoF | Muscles |
|---|---|---|
| `Mjlab-SuperMimic-Flat-MS-Human-700` | 37 + free root | 700 |
| `Mjlab-SuperMimic-Flat-MS-Human-700-Locked-Hands` | 33 + free root | 694 |

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
```

Runs log to `logs/rsl_rl/mshuman700_tracking/<date>/` (`train_log.csv`, tensorboard, `model_*.pt` every
2000 iterations). mjlab's CLI writes tuples in Python syntax: `--agent.wm-hidden "(1024,1024,1024)"`.

Memory with the defaults, roughly: replay buffer ~14 GB (4096 x 1000 transitions, 700-D activations
and excitations in fp16), imagination BPTT (4096 x 32 steps) a few tens of GB. `--agent.buffer-steps`,
`--agent.pi-batch` and `--env.scene.num-envs` are the knobs.

## What is the same as supermimic_vibed

* **Observation** (decoder, 10-frame history; plus the activations, see below): q - q0, q̇, root angular velocity (IMU), projected
  gravity, a_{t-1}, reference root orientation relative to the robot (6D). **Goal** (encoder): K=10
  future reference frames (root displacement and orientation in the heading frame, root velocities in
  the root frame, q, 0.1 q̇).
* **Reward** (x dt): BeyondMimic tracking on 17 bodies (body pos 0.3 / ori 0.4 / lin vel 1.0 / ang vel
  3.14, root pos 0.3 and ori 0.4 at weight 0.5, reference re-anchored to the robot's xy + yaw); no
  self-collision term, and (see below) no action-rate or joint-limit penalty.
* **Termination**: |e_z| > 0.25 m or tilt error > 0.8 (sigmoid-softened in imagination; plus MuscleMimic's deviation checks, see below); time-outs at
  10 s and at the clip end, bootstrapped.
* **Reset sampling**: 1 s bins, 50% uniform / 40% softmax(-25 G_b / G_max) / 10% its running average.
* **World model** MLP(2048,2048,2048) + LayerNorm + SiLU on heading-invariant features, frame-local
  increments, k=8-step autoregressive loss with per-component error floors; **policy** encoder
  MLP(2048,1024,512,256) → z∈R^32 (tanh) + decoder MLP(2048,1024,512,256); **critic** MLP(256,128) on
  GAE(λ=0.95) targets, 5 epochs x 4 minibatches; BPTT over H=32 imagined steps with soft survival and
  critic bootstrap; Adam 2e-4 (policy β=(0.9,0.95)), cosine decay to 0.03x over 250k iterations, grad
  clip 1.0 (+0.1 element-wise for the policy), γ=0.99, 500 world-model-only iterations, bf16 MLPs.
* **Evaluation**: App. F.1 as in supermimic_vibed.

## Deviations from supermimic_vibed

Required by the muscle-actuated model:

* **Action** = 700 muscle excitations. The decoder outputs logits, `e = sigmoid(logits)` is the env
  action (initialized at e = 0.05); exploration noise N(0, 0.1²) is added to e and clipped to [0, 1], as
  supermimic adds it to the action (on the logits it would shrink to 0.1 e(1-e) at low e). The a_{t-1}
  observation is on e.
* **Muscle activations are state.** The plant's 700 activations (time constants 10/40 ms) are part of
  the state; their dynamics are known exactly, so they are integrated analytically with MuJoCo's own
  substeps (matches the simulator to float precision) instead of being learned. The world model
  predicts the body state only; its action input is the substep-mean activation of the step. The
  critic sees the activations too.
* **Kinematics**: 37 independent joints; the 42 coupled joints come from the `<equality>` polynomials
  (in FK and when writing reset states). The torch FK replicates `mj_kinematics` (multi-joint bodies,
  off-origin joints, welded bodies) and matches MuJoCo to 1e-9.
* **Root**: one free joint instead of 6 OpenSim DoF (no gimbal lock), canonical z-up root frame.
* **Tracked bodies**: pelvis, femur, tibia, calcn, toes (x2), thoracic12, head_neck, humerus, ulna,
  hand (x2) — the G1 set mapped onto the skeleton, plus the toes.
* **Simulator**: mjlab / MuJoCo-Warp, 2 ms x 10 substeps (50 Hz, the model's native timestep),
  implicitfast, Newton 10 iterations; robot-robot contacts disabled (as in supermimic_vibed). Upstream,
  the arms have no collision geoms, so hands and arms pass through the ground.

Taken from MuscleMimic, where SuperMimic trains stably on a muscle model (supermusclemimic), after the
runs here kept losing the critic:

* **Observation**: the decoder also sees the current 700 activations (no history), as MuscleMimic's
  policy does; otherwise it has to infer them from the 10-step excitation history.
* **Termination**: MuscleMimic's deviation checks on top of the above -- mean root-relative body
  position error > 1 m, root position error > 1 m (catches horizontal drift), root orientation error
  > 1 rad (heading included), the thresholds of supermusclemimic's runs; softened with 5 % widths.
* **Reward**: no action-rate (-0.1) or joint-limit (-2) penalty, as in MuscleMimic (both coefficients
  0). Summed over ~620 muscles (G1: 29 actuators) and over the joints a collapsing body pins at their
  limits, they were as large as the tracking reward: the net step reward sat at zero or below, staying
  alive was worth nothing (imagined survival never rose) and every critic diverged.

Optional, off by default (`--agent.critic-target-alpha`, `--agent.value-clip`): a SHAC-style target
critic and clamping bootstrapped values to the attainable maximum.

By request, for now:

* **No randomization of any kind**: no friction / encoder offset / CoM domain randomization (the sim
  context ξ is removed from the world model and critic), no observation noise, no reset noise. Exploration
  noise on the actions and the sampling of reset frames are part of the algorithm and stay.

## Attribution

* Method: Q. Liao et al., *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World
  Model*, CoRL 2026, arXiv:2610.28480. Implementation ported from supermimic_vibed by Chong Zhang (MIT,
  Copyright (c) 2026 ETH Zurich, Chong Zhang); ported files say so in their docstrings.
* MS-Human-700 (Apache-2.0, LNSGroup), msgym walking data (Apache-2.0), mjlab (Apache-2.0).
