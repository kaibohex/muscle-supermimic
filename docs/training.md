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

## Commands

```bash
# train (logs to logs/rsl_rl/<robot>_tracking/<date>_<run name>/: train_log.csv, tensorboard, model_*.pt)
uv run train Mjlab-SuperMimic-Flat-MyoFullBody --env.commands.motion.motion-file data/musclemimic/kit_kinesis_train.txt
# evaluate a run's checkpoints as they appear (CPU), or one checkpoint, recording the rollout
uv run ms-eval --task Mjlab-SuperMimic-Flat-MyoFullBody --run-dir logs/rsl_rl/myofullbody_tracking/<run> \
    --motion-file data/musclemimic/kit_kinesis_test.txt --device cpu --watch
uv run ms-eval --task Mjlab-SuperMimic-Flat-MyoFullBody --checkpoint <model.pt> --motion-file <clips> --record rollout.npz
# render a recorded rollout: the simulated body, each muscle colored by its activation (light grey 0 to deep red 1),
# studio look, 1920x1080 at 50 fps (--ghost overlays the reference, --no-activation keeps the model's muscle color)
uv run ms-render rollout.npz videos/
# watch a policy in mjlab's viewer
uv run play Mjlab-SuperMimic-Flat-MyoFullBody --checkpoint-file <model.pt> --motion-file <clip.npz>
```

mjlab's CLI writes tuples in Python syntax: `--agent.wm-hidden "(1024,1024,1024)"`.

## Solver

The tasks simulate with the Euler integrator and 100 / 50 solver / line-search iterations, as MuscleMimic
simulates MyoFullBody (`--env.sim.mujoco.integrator`, `.iterations`, `.ls-iterations`; mjlab's default is
implicitfast at 10 / 20). It costs nothing: physics is under 10 % of an iteration, and Euler at 100 / 50 steps
as fast as implicitfast at 10 / 20 (0.095 vs 0.107 s per simulator step, 4096 MyoFullBody envs, RTX 4090 D).
Our MyoFullBody run with Euler took off around iteration 3500; one with implicitfast had not by iteration 6600
(one seed each). Training, evaluation and `play` all use these settings.

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

**Compilation on a new machine.** The runner's compiled pieces and CUDA graphs cost a cold machine about 9 minutes
before the first policy update (trial t17: the first capture at 10.6 minutes of the training clock instead of about
1), which a `--agent.max-hours` budget counts. `scripts/compile_cache.sh save <name>` packs a warm machine's caches
(torch Inductor / AOTAutograd / Triton, Warp's kernels, the CUDA JIT cache) to the results dataset on Hugging Face;
`COMPILE_CACHE=<name>` in `skypilot/train_blink.yaml` / `scripts/run_trial.sh` loads them before training, outside
its clock. Save them from a machine that has trained the same configuration with the same code and lockfile: the
caches key on all three.

## BlinkMimic recipe

[BlinkMimic](https://github.com/fuyichen0725/blinkmimic) (Fuyi Chen, 2026; built on InstantTrack and
supermimic_vibed) trains the Unitree G1 on 40 LAFAN1 clips to a normalized return of 0.82 in about 3 minutes on
one RTX 4090, from SuperMimic's ~40: about 6x fewer iterations and 2.3x less time per iteration. Its changes are
being ported here as runner flags, every one off by default, so `uv run train` without them is the runner above.

**Current recipe and result (2026-11-05).** `scripts/train_blink.sh` sets the table below plus 1024 environments,
both CUDA-graph options with the collection overlapped (`--agent.overlap-collect True`), no imagination
rematerialization and Beta exploration noise (`--agent.action-noise-dist beta`), at a 1x policy learning rate. On
MS-Human-700 with locked hands and no muscle observations, one RTX 5090 with warm compile caches, two seeds reach a
held-out return of 0.83 at 8.5 minutes, 0.90 at 10-11 minutes and track all 108 test clips from about 9.5 minutes
(trials 30 and 31; the published baseline: 0.83 after 4.3 hours). A cold machine adds about 9.5 minutes of
compilation unless it loads a saved compile cache (below). A higher learning rate (a 2x policy phase, or a 6e-4
base rate) stalls about half of the runs at 0.3-0.4, so keep 1x; decide recipe changes on at least two seeds.

**Ported so far** (`scripts/train_blink.sh <task> <motion-file> [train flags]` sets all of it):

| change | flag | from |
|---|---|---|
| Polyak average of the policy as the deployed policy (checkpoint key `pi`; the live policy is `pi_live`) | `--agent.pi-ema 0.995` | InstantTrack |
| Muon on the hidden weight matrices of policy and world model, Adam elsewhere, at Adam's update RMS | `--agent.pi-opt muon --agent.wm-opt muon` | BlinkMimic |
| no host synchronization in the update loop: normalizer freeze, start sampling and the finite-gradient gate on the device | `--agent.no-sync True --agent.fused-adam True` | InstantTrack |
| learning rate 4e-4, world model 3 x 1024 with 5 batches per iteration | `--agent.lr 4e-4 --agent.wm-hidden "(1024,1024,1024)" --agent.wm-batches 5` | InstantTrack |
| no world-model-only warm-up | `--agent.wm-only-iters 0` | BlinkMimic |
| imagination batch 2048 with a 4x policy learning rate, decayed to 1x over 300 iterations once the training return reaches 0.75, at iteration 1400 at the latest | `--agent.pi-batch 2048 --agent.pi-lr-mult 4 --agent.lr-decay-ret 0.75 --agent.lr-decay-len 300 --agent.lr-decay-by 1400` | BlinkMimic |
| reset statistics: the episodes of a bin averaged before its EMA step | `--env.commands.motion.bin-fix True` | BlinkMimic |
| 30 % of the resets one bin before recent termination points | `--env.commands.motion.fail-reset 0.3` | InstantTrack |
| no `torch.cuda.empty_cache()` per iteration | `--agent.empty-cache False` | |
| semi-implicit world model: velocity increments, positions integrated from the old and new velocities as the Euler substeps do (weight 0.6 on the new one for 5 substeps), plus predicted corrections | `--agent.wm-structured True` | BlinkMimic |
| left-right mirror image of half of each world-model batch (states, activations, excitations, touch) | `--agent.wm-mirror 0.5` | BlinkMimic |
| evaluation-guided resets: from iteration 1000, every 125 iterations the deployed policy is swept over all clips under evaluation conditions, one probe world per 6 s tile of each clip started 4-9 s earlier on the reference; 15 % of the resets go to the 10 s before the places where it fails (counts decaying by 0.7 per sweep) | `--agent.eprobe-share 0.15` (`--agent.eprobe-*`) | BlinkMimic |
| the world-model update and the policy + critic update as CUDA graphs | `--agent.cuda-graph True` | InstantTrack / BlinkMimic |
| one collection step, episode ends and resets included, as one CUDA graph through a thin simulator wrapper | `--agent.collect-graph True` | BlinkMimic (`--collect_graph --reset_masked`) |
| the collection on a second CUDA stream, concurrently with the updates (its own policy copy, normalizers, random state and capture stream; the updates sample the buffer as it stood at the start of the iteration) | `--agent.overlap-collect True` | |
| stop after a wall-clock budget, with a final checkpoint | `--agent.max-hours H` | |
| exploration noise from a Beta distribution with the policy's excitation as its mean (standard deviation `action_noise` at 0.5), instead of a clipped Gaussian | `--agent.action-noise-dist beta` | |
| exploration noise `N(0, action_noise^2)` folded back into [0, 1] at the bounds instead of clipped | `--agent.action-noise-dist reflect` | |

`train_log.csv` gains `pi_lr`, `pi_mult` and `nonfinite` (skipped non-finite policy updates). `ms-eval` and `play`
read the deployed policy; resuming restores the average, the trigger and the counters.

**Exploration noise and the excitation box.** The default noise, `a + N(0, 0.1^2)` clipped to [0, 1], is not
mean-preserving where the policy keeps muscles nearly silent. On a trained MS-Human-700 policy (Trial 7, 512 envs,
150 steps) 26 % of the excitations are below 0.05, 17.9 % of the noisy ones leave the box (14.9 % below 0), and for
the near-silent quarter the executed excitation averages 0.050 against the intended 0.0185: the noise adds a tonic
co-contraction the deterministic policy never produces. `--agent.action-noise-dist beta` draws from
`Beta(k a, k (1 - a))`, `k = 1 / (4 sigma^2) - 1`, whose mean is the excitation itself and whose spread shrinks
towards the bounds (`networks.explore`). That spread is the catch: any noise that stays in [0, 1] with mean `a` has a
standard deviation of at most `sqrt(a (1 - a))`, and fixed-`k` Beta keeps a fifth of it, 0.028 at a = 0.02 against
0.065 for the clipped Gaussian; only 1.4 % of its draws lift a near-silent muscle by more than 0.1 (clipped Gaussian:
16 %). Trial 10 stalled at 0.40 held-out with it (the world model rarely sees those muscles act). `reflect` keeps the
Gaussian's step sizes and folds the draws back into the box instead of piling 42 % of them on 0; it is biased upwards
like clipping, more so (0.081 at a = 0.02), and lifts 24 % of the near-silent draws by more than 0.1.

**What the first runs showed** (MS-Human-700 with locked hands, 4096 envs, one RunPod RTX 5090, 2026-11-03;
`experiments/trials.json` and its page have the records). Three of BlinkMimic's G1 settings do not transfer, and
`scripts/train_blink.sh` sets the recipe without them:

| BlinkMimic setting | on 694 muscles | the preset |
|---|---|---|
| no world-model warm-up, bootstrapped values unclamped | the critic diverges from the first policy updates (loss 195, imagined return 170 where 14 is attainable) | `--agent.value-clip True --agent.wm-only-iters 200` |
| 4x policy learning rate until the return reaches 0.75 or iteration 1400 | 2x ahead of the baseline to iteration 1000, then the policy degrades (gradient-clip fraction from 0 to 0.13) and does not recover | `--agent.pi-lr-mult 1` |
| evaluation-guided resets every 125 iterations, 10 s windows | every probe fails within 3 s on 5 s clips, so the focus covers whole clips; the eager sweep costs half the training time; not the cause of the degradation (tested with them off) | `--agent.eprobe-share 0` |

With the rest of the recipe (Trial 5): held-out return 0.53 at 1.0 h and 0.80 with 94 of 108 clips tracked at
1.52 h (iteration 3000), where the published baseline has 0.26 at 1.09 h and 0.61 at iteration 3000, and first
reaches 0.82 at iteration 5500 (3.3 h); 1.80 s per iteration against the baseline's 2.38.

**Time per iteration.** `--agent.cuda-graph True` captures the world-model update and the policy + critic update as
CUDA graphs after three warm-up calls each and replays them (the same updates up to rounding, another random
stream; needs `--agent.no-sync True --agent.fused-adam True --agent.empty-cache False`). `--agent.collect-graph True`
does the same for the collection step: `rl/supermimic/collector.py` replaces mjlab's `env.step` and `env.reset(ids)`
during collection with a thin wrapper on the same tensors (the analytic observation, the policy and its noise
written as the muscles' `ctrl`, MuJoCo-Warp's substeps with the solver loop, rewards and terminations from the
analytic heads, the buffer write at a device-side write head, the normalizers, and the episode ends as masked
updates: the bin statistics, MuJoCo-Warp's masked `reset_data`, an inverse-CDF reset draw for every env written where
an episode ended, two `forward` passes), captured with Warp's capture nested in torch's after three eager steps.
`tests/test_collector.py` checks it against the eager path: equal to 1e-5 on the CPU over 40 steps with failures and
time-outs, within MuJoCo-Warp's run-to-run noise on the GPU. It keeps one quirk of the eager path for parity: mjlab's
contact-sensor cache is filled before the step's `forward` and dropped only by a reset, so the buffer's touch after a
step is the touch at the start of the last substep, while the next observation reads the post-`forward` touch
whenever some env reset in that step (at thousands of envs nearly always). It needs `--agent.no-sync True`,
`--env.commands.motion.bin-fix True`, no evaluation-guided resets and no observation history; the probe sweep still
runs eagerly. The `-No-Muscle-Obs` task variants (docs/mdp.md) leave the
muscle length, velocity and force out of the observation, which removes the muscle geometry, the largest cost of an
imagined step, from both sides: on MyoFullBody the policy update halves, and the learning curve was the same in a
run to 16000 iterations.

The mirror maps (`tasks/tracking/mdp/mirror.py`): bodies, muscles and touch values pair by name (`_r` / `_l`,
`_R` / `_L`, a bare name and its `_left` / `_L` twin), each joint's partner and sign come from the forward
kinematics (a joint moved by a small angle, mirrored, must equal its partner moved by plus or minus that angle), and
every map is validated on random poses when the runner starts, through the kinematics and the muscle geometry. The
residuals measure the model's own asymmetry: MS-Human-700's skeleton is symmetric to float precision, MyoFullBody's
to 2 mm (arms, head, toes); muscle paths to well under 1 mm, except LTpT_T12 (3-4 cm on both models, an upstream
path quirk), EDCL (1.3 cm) and two multifidus segments (2-6 mm) on MS-Human-700, DELT3 and PQ (1-3 mm) on
MyoFullBody. The runner prints them; a model asymmetric beyond `Mirror.TOL` raises instead of poisoning half of
every batch.

The probe sweep runs eagerly in a second env built at the first sweep (`eprobe: N probe worlds ...` on the console;
on MuscleMimic's KIT training set about 1100 worlds, since most clips are shorter than one tile and start at
their first frame). Each sweep prints its steps, time and the failed probes and clips, and `train_log.csv` gains
`probe_fail`, the last sweep's failure fraction. A sweep costs as much as its steps of that many worlds, up to
15 s of simulation, so with the default interval it is a sizeable share of the training time until the sweep runs
as a CUDA graph on a side stream (BlinkMimic's `--eprobe_graph --eprobe_async`, not ported); `--agent.eprobe-every`
trades the focus's freshness for that cost.

**Not ported (yet)**: the CUDA graph of the probe sweep (the collector's wrapper is the starting point), light
background checkpoints, and BlinkMimic's evaluation tooling
(replicas, time to target, accuracy metrics). BlinkMimic's reference-action skip does not apply: the actions here
are excitations, with no reference excitation to build on.

`scripts/bench_iteration.py` times one iteration piece by piece (collection, world-model updates, policy update)
for any flag set, e.g. `--set pi_batch=2048 pi_opt=muon no_sync=true fused_adam=true --env-set bin_fix=true`.
On one RTX 5090 (MS-Human-700 locked hands without muscle observations, 4096 envs, the recipe, 2026-11-04):

| configuration | collection | world model | policy + critic | s / iteration |
|---|---|---|---|---|
| eager | 0.465 | 0.054 | 0.754 | 1.27 |
| `--agent.cuda-graph True` | 0.471 | 0.025 | 0.151 | 0.65 |
| plus `--agent.collect-graph True` | 0.310 | 0.025 | 0.151 | 0.49 |

`scripts/bench_collect.sh` reruns the last row on a training machine. Collection is then mostly MuJoCo-Warp itself
(seven solver passes of 4096 worlds per step: five substeps and two `forward`s).
