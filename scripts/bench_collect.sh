#!/bin/bash
# The t7 configuration on one GPU (MS-Human-700 locked hands without muscle observations, 4096 envs, the BlinkMimic
# recipe with both CUDA-graph options): seconds per iteration of collection and updates (scripts/bench_iteration.py).
# Usage, on the training machine:  bash scripts/bench_collect.sh [collect_graph values, default "true"]
# NUM_ENVS (default 4096), SIM_SET (MuJoCo solver overrides, e.g. "iterations=50 ls_iterations=25") and EXTRA_SET (runner
# overrides, e.g. "overlap_collect=true"), ENV_SET (motion command overrides, e.g. "compile=false") and PROFILE (a number
# of extra profiled iterations, run without CUDA graphs: the top CUDA kernels) change the setup.
set -u
cd ~/sky_workdir
export MUJOCO_GL=disable PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SETS="pi_batch=2048 wm_batches=5 wm_hidden=1024,1024,1024 pi_opt=muon wm_opt=muon fused_adam=true no_sync=true pi_ema=0.995 empty_cache=false wm_structured=true wm_mirror=0.5 value_clip=true lr=4e-4 cuda_graph=true"
for CG in ${1:-true}; do
  echo "=== collect_graph=$CG num_envs=${NUM_ENVS:-4096} sim_set=${SIM_SET:-default} extra=${EXTRA_SET:-none} env=${ENV_SET:-none} $(date -u +%T) ==="
  .venv/bin/python scripts/bench_iteration.py --task Mjlab-SuperMimic-Flat-MS-Human-700-Locked-Hands-No-Muscle-Obs \
    --motion-file data/mshuman700_kit/kit_kinesis_train.txt --num-envs ${NUM_ENVS:-4096} --warmup 8 --iters 20 \
    --env-set bin_fix=true fail_reset=0.3 ${ENV_SET:-} --profile ${PROFILE:-0} --sim-set ${SIM_SET:-} --set $SETS collect_graph=$CG ${EXTRA_SET:-} 2>&1 \
    | grep -v "Module \|FutureWarning\|warnings.warn\|^\[INFO\]\||\|^+--" | tail -${TAIL:-12}
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
done
echo "=== bench done $(date -u +%T) ==="
