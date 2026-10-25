# Training

## Algorithm

The SuperMimic runner (`rl/supermimic/`) follows the unofficial reproduction
[supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) (MIT), with the choices SuperMimic's paper
leaves open for muscles (reward scale, critic input, exploration noise on the excitations, activations as
world-model inputs) set as in a SuperMimic port to MuscleMimic in which training is stable.

* **Reset sampling**: 1 s bins, 50% uniform / 40% softmax(-25 G_b / G_max) / 10% its running average.
* **World model** MLP(2048,2048,2048) + LayerNorm + SiLU on heading-invariant features and the
  activations before and after the step, frame-local increments plus a foot-touch head, k=8-step
  autoregressive loss with per-component error floors (+ touch MSE).
* **Policy** encoder MLP(2048,1024,512,256) → z∈R^32 (tanh) + decoder MLP(2048,1024,512,256) →
  excitations e = sigmoid(·) in [0, 1], initialized at 0.1; exploration noise N(0, 0.1²) on e, clipped.
* **Critic** MLP(256,128) on the decoder observation, the goal and privileged reference errors (root
  position in the heading frame, heading error, joint position and velocity errors), GAE(λ=0.95)
  targets, 5 epochs x 4 minibatches.
* **Policy update**: BPTT over H=32 imagined steps (0.32 s) with soft survival and critic bootstrap, each
  step recomputed in the backward pass; Adam 2e-4 (policy β=(0.9,0.95)), cosine decay to 0.03x over 250k
  iterations, grad clip 1.0 (+0.1 element-wise for the policy), γ=0.99, 500 world-model-only iterations,
  bf16 MLPs, 4096 envs, imagination batch 4096, 2.5 simulator steps per iteration (UTD 0.4).
* **Evaluation** (`ms-eval`): every clip once from its first frame, as App. F.1 of the paper and
  supermimic_vibed; normalized return and failed clips. `--action-noise 0.1` evaluates with the training noise.

Optional, off by default (`--agent.critic-target-alpha`, `--agent.value-clip`): a SHAC-style target critic
and clamping bootstrapped values to the attainable maximum.

`--agent.state-grad-clip <c>` (off by default, not in SuperMimic): at every imagined step, each trajectory's
gradient with respect to what it carries to the next step (body state, activations, foot touch, last action) is
clipped to norm c, measured on that trajectory's own return; `pi_state_clip` in the log is the fraction of
trajectory steps clipped. It bounds the gradient growth around the loop muscle-velocity observation → policy →
activation → world model, which otherwise lets 1 % of the imagined trajectories dominate an update (see the
[project page](website/index.html)). We use c = 100; per-step norms of the other trajectories have a median of ~6 and a
99th percentile of ~130. With it, MS-Human-700 takes off around iteration 2000 in two seeds (without it, not by
6000), and MyoFullBody reaches 166-step episodes by iteration 2000 (82 without); 0.1-0.2 % of trajectory steps get
clipped.

## Commands

```bash
# train (logs to logs/rsl_rl/<robot>_tracking/<date>_<run name>/: train_log.csv, tensorboard, model_*.pt)
uv run train Mjlab-SuperMimic-Flat-MyoFullBody --env.commands.motion.motion-file data/musclemimic/kit_kinesis_train.txt \
    --env.sim.mujoco.integrator euler --env.sim.mujoco.iterations 100 --env.sim.mujoco.ls-iterations 50
# evaluate a run's checkpoints as they appear (CPU), or one checkpoint, recording the rollout
uv run ms-eval --task Mjlab-SuperMimic-Flat-MyoFullBody --run-dir logs/rsl_rl/myofullbody_tracking/<run> \
    --motion-file data/musclemimic/kit_kinesis_test.txt --device cpu --watch
uv run ms-eval --task Mjlab-SuperMimic-Flat-MyoFullBody --checkpoint <model.pt> --motion-file <clips> --record rollout.npz
# render a recorded rollout: the simulated body with its muscles, studio look, 50 fps (--ghost overlays the reference)
uv run ms-render rollout.npz videos/
# watch a policy in mjlab's viewer
uv run play Mjlab-SuperMimic-Flat-MyoFullBody --checkpoint-file <model.pt> --motion-file <clip.npz>
```

mjlab's CLI writes tuples in Python syntax: `--agent.wm-hidden "(1024,1024,1024)"`.

## Solver

Train with the Euler integrator and 100 / 50 solver / line-search iterations, as MuscleMimic simulates
MyoFullBody (the flags above). It costs nothing: physics is under 10 % of an iteration, and Euler at
100 / 50 steps as fast as mjlab's default, implicitfast, at 10 / 20 (0.095 vs 0.107 s per simulator step,
4096 MyoFullBody envs, RTX 4090 D). Our MyoFullBody run with Euler took off around iteration 3500; one with
implicitfast had not by iteration 6600 (one seed each).

## Memory and speed

| | torch peak (4096 envs) | runs on | iterations / s |
|---|---|---|---|
| MyoFullBody | 15-17 GB | 24 GB (RTX 4090 D) | 0.31 (RTX 4090 D, RTX 5090) |
| MS-Human-700, locked hands | 23 GB | 32 GB (RTX 5090) | 0.42 (RTX 5090) |

The world-model-only iterations at the start run 3-4x faster. MuJoCo-Warp allocates outside torch's pool,
so the card needs several GB beyond torch's peak. `--env.scene.num-envs`, `--agent.buffer-steps` and
`--agent.pi-batch` are the memory knobs:

```bash
uv run train Mjlab-SuperMimic-Flat-MS-Human-700-Locked-Hands --env.commands.motion.motion-file data/mshuman700_kit/kit_kinesis_train.txt \
    --env.scene.num-envs 1024 --agent.buffer-steps 500 --agent.pi-batch 2048
```

**RTX 50xx (sm_120)**: Triton 3.8 fails to compile some of the fused muscle-wrap kernels on these GPUs, so
the simulator-side pieces stay eager there (`MotionCommandCfg.compile` is turned off automatically; the
runner's own compiled pieces are unaffected).
