#!/bin/bash
# Train with the BlinkMimic recipe as it works on the muscle models (docs/training.md, "BlinkMimic recipe"): the policy
# EMA, Muon on the hidden matrices of policy and world model, no host synchronization in the update loop, fused Adam,
# learning rate 4e-4, a 3 x 1024 world model with 5 batches per iteration, an imagination batch of 2048, the
# reset-statistics fix and 30 % of the resets before recent failure points, the semi-implicit world model and
# left-right mirror augmentation of half of each world-model batch. Three of BlinkMimic's G1 settings are not used: the
# bootstrapped values are clamped and the world model warms up for 200 iterations (without both the critic diverges on
# 694 muscles), the policy learning rate stays at 1x (the 4x phase wrecked the policy after iteration 1000), and the
# evaluation-guided resets are off (every probe fails within 3 s of its start on 5 s clips, and the eager sweep costs
# half the training time).
#
# Since 2026-11-05 the preset also carries the systems and exploration settings of the fastest runs (Trials 30 and 31
# of experiments/trials.json, MS-Human-700 locked hands without muscle observations, one RTX 5090 with warm compile
# caches): 1024 environments, the update and collection steps as CUDA graphs with the collection overlapped with the
# updates, no rematerialization of the imagined steps, and Beta exploration noise (mean the policy's excitation, never
# outside [0, 1]). First 0.83 held-out return at 8.5 minutes on both seeds (the published baseline: 4.3 hours). Keep
# the policy learning rate at 1x: a 2x phase or a 6e-4 base rate stalls about half of the runs at 0.3-0.4.
#
# Since 2026-11-06 the policy acts at 50 Hz (Trials 45 and 46): 10 physics substeps per control step instead of 5, every
# step-counted setting rescaled to keep its duration (goal frames 10 steps = 0.2 s apart, 16 imagined steps = 0.32 s,
# gamma 0.98 and lambda 0.90, a 500-step = 10 s buffer, 4-step = 80 ms world-model windows, 1.25 simulator steps =
# 25 ms of motion per iteration), and the exploration noise scaled by 1/sqrt(2) to 0.0707, so that the variance injected
# per second of motion is the 100 Hz recipe's. First 0.83 held-out at 6.4 minutes on both seeds (100 Hz: 8.9 / 9.0),
# best 0.944 / 0.945. Evaluate these policies at the same rate: ms-eval --decimation 10 --stride 10 (run_trial.sh's
# EVAL_ARGS default); the compile cache for it is ms700_1024_beta_50hz. For 100 Hz append --env.decimation 5
# --env.commands.motion.stride 20 --agent.horizon 32 --agent.gamma 0.99 --agent.lam 0.95 --agent.buffer-steps 1000
# --agent.wm-window 8 --agent.utd 0.4 --agent.action-noise 0.1.
#
# Usage: bash scripts/train_blink.sh <task> <motion-file> [more train flags, e.g. --agent.run-name blink_s0 --agent.seed 1]
#   bash scripts/train_blink.sh Mjlab-SuperMimic-Flat-MS-Human-700-Locked-Hands data/mshuman700_kit/kit_kinesis_train.txt --agent.run-name blink
# Later flags override earlier ones, so e.g. --agent.pi-batch 4096 after the preset restores that setting. mjlab's CLI
# takes booleans as values (--agent.no-sync True) and tuples in Python syntax. TRAIN names the train command (default
# "uv run train"; a cloud job sets ".venv/bin/python -m mjlab.scripts.train"). The CUDA graphs need a GPU: on a CPU add
# --agent.cuda-graph False --agent.collect-graph False --agent.overlap-collect False.
set -eu
cd "$(dirname "$0")/.."
TASK=${1:?usage: bash scripts/train_blink.sh <task> <motion-file> [train flags]}
MOTION=${2:?usage: bash scripts/train_blink.sh <task> <motion-file> [train flags]}
shift 2
BLINK=(
  --agent.pi-ema 0.995 --agent.no-sync True --agent.fused-adam True --agent.lr 4e-4
  --agent.wm-batches 5 --agent.wm-hidden "(1024,1024,1024)"
  --agent.pi-opt muon --agent.wm-opt muon --agent.wm-only-iters 200 --agent.value-clip True
  --agent.pi-batch 2048 --agent.pi-lr-mult 1
  --agent.empty-cache False
  --agent.wm-structured True --agent.wm-mirror 0.5
  --env.scene.num-envs 1024 --agent.cuda-graph True --agent.collect-graph True --agent.overlap-collect True
  --agent.remat-imagination False --agent.action-noise-dist beta
  --agent.eprobe-share 0
  --env.commands.motion.bin-fix True --env.commands.motion.fail-reset 0.3
  --agent.save-interval 250 --agent.max-iterations 20000
  --env.decimation 10 --env.commands.motion.stride 10 --agent.horizon 16 --agent.gamma 0.98 --agent.lam 0.9
  --agent.buffer-steps 500 --agent.wm-window 4 --agent.utd 0.8 --agent.action-noise 0.0707
)
exec ${TRAIN:-uv run train} "$TASK" --env.commands.motion.motion-file "$MOTION" "${BLINK[@]}" "$@"
