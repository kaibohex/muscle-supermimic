#!/bin/bash
# Train with the BlinkMimic recipe as it works on the muscle models (docs/training.md, "BlinkMimic recipe"): the policy
# EMA, Muon on the hidden matrices of policy and world model, no host synchronization in the update loop, fused Adam,
# learning rate 4e-4, a 3 x 1024 world model with 5 batches per iteration, an imagination batch of 2048, the
# reset-statistics fix and 30 % of the resets before recent failure points, the semi-implicit world model and
# left-right mirror augmentation of half of each world-model batch. Three of BlinkMimic's G1 settings are not used: the
# bootstrapped values are clamped and the world model warms up for 200 iterations (without both the critic diverges on
# 694 muscles), the policy learning rate stays at 1x (the 4x phase wrecked the policy after iteration 1000), and the
# evaluation-guided resets are off (every probe fails within 3 s of its start on 5 s clips, and the eager sweep costs
# half the training time). This is Trial 5 of experiments/trials.json: 0.80 held-out return at 1.5 h on one RTX 5090,
# where the published baseline has 0.61.
#
# Usage: bash scripts/train_blink.sh <task> <motion-file> [more train flags, e.g. --agent.run-name blink_s0 --agent.seed 1]
#   bash scripts/train_blink.sh Mjlab-SuperMimic-Flat-MS-Human-700-Locked-Hands data/mshuman700_kit/kit_kinesis_train.txt --agent.run-name blink
# Later flags override earlier ones, so e.g. --agent.pi-batch 4096 after the preset restores that setting. mjlab's CLI
# takes booleans as values (--agent.no-sync True) and tuples in Python syntax. TRAIN names the train command (default
# "uv run train"; a cloud job sets ".venv/bin/python -m mjlab.scripts.train"). --agent.cuda-graph True adds the CUDA
# graphs of the updates (docs/training.md).
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
  --agent.eprobe-share 0
  --env.commands.motion.bin-fix True --env.commands.motion.fail-reset 0.3
  --agent.save-interval 250 --agent.max-iterations 20000
)
exec ${TRAIN:-uv run train} "$TASK" --env.commands.motion.motion-file "$MOTION" "${BLINK[@]}" "$@"
