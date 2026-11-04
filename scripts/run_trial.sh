#!/bin/bash
# One training run and its evaluation, then the upload to the HF dataset: the run section of skypilot/train_blink.yaml,
# also started by hand on a machine the task already set up (when the local SkyPilot API server is unavailable):
#   ssh <cluster> 'cd sky_workdir; read -r HF_TOKEN; export HF_TOKEN RUN_NAME=... TASK=... EVAL_DURING=0 EXTRA_ARGS="...";
#                  setsid nohup bash scripts/run_trial.sh > logs/job_$RUN_NAME.log 2>&1 &' < ~/.cache/huggingface/token
# The variables and their defaults are the task's (skypilot/train_blink.yaml, envs).
: "${RUN_NAME:=blink_ms700_s0}" "${TASK:=Mjlab-SuperMimic-Flat-MS-Human-700-Locked-Hands}"
: "${MOTION:=data/mshuman700_kit/kit_kinesis_train.txt}" "${TEST_MOTION:=data/mshuman700_kit/kit_kinesis_test.txt}"
: "${EXPERIMENT:=mshuman700_tracking}" "${EXTRA_ARGS:=--agent.max-iterations 8000 --agent.seed 0}"
: "${KEEP_CKPTS:=model_002000.pt model_004000.pt model_006000.pt model_008000.pt}" "${EVAL_DURING:=1}"
: "${HF_RESULTS_REPO:=YOUR_HF_USER/supermimic-runs}" "${PREWARM:=0}"
set -uo pipefail
cd ~/sky_workdir
export PATH=$HOME/.local/bin:$PATH MUJOCO_GL=disable PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRAIN=".venv/bin/python -m mjlab.scripts.train"
mkdir -p logs
# the held-out evaluation of every checkpoint as it appears, on the same GPU (an env of one world per test clip),
# started once the run directory exists
EVAL_PID=""
if [ "$EVAL_DURING" = "1" ]; then
  ( sleep 300
    RUN=$(ls -dt logs/rsl_rl/$EXPERIMENT/*_$RUN_NAME 2>/dev/null | head -1)
    [ -n "$RUN" ] && exec .venv/bin/python -m muscle_supermimic.tasks.tracking.scripts.evaluate --task $TASK --run-dir $RUN \
      --motion-file $TEST_MOTION --device cuda:0 --watch > logs/eval_$RUN_NAME.out 2>&1
  ) &
  EVAL_PID=$!
fi
if [ "${PREWARM:-0}" = "1" ]; then
  # a fresh machine compiles for ~14 minutes on the first iterations, which would count against a wall-clock budget
  # (--agent.max-hours): fill the compile caches first with the benchmark of the same configuration
  echo "== prewarm $(date -u +%FT%TZ)"
  bash scripts/bench_collect.sh true 2>&1 | grep -E "^collect |^total |==="
fi
echo "== train $RUN_NAME $(date -u +%FT%TZ)"
bash scripts/train_blink.sh $TASK $MOTION --agent.run-name $RUN_NAME $EXTRA_ARGS 2>&1 \
  | tee logs/train_$RUN_NAME.out | grep --line-buffered -iE "^\s*it [0-9]*[05]00 |^eprobe|^wm_mirror|^lr_decay|^collect_graph|^max_hours|error|traceback|missing|usage|loading|cuda"
TRAIN_STATUS=${PIPESTATUS[0]}
echo "== train exit $TRAIN_STATUS $(date -u +%FT%TZ)"
grep -E "^\s*it " logs/train_$RUN_NAME.out | tail -1
RUN=$(ls -dt logs/rsl_rl/$EXPERIMENT/*_$RUN_NAME 2>/dev/null | head -1)
if [ -n "$RUN" ]; then
  if [ -n "$EVAL_PID" ]; then sleep 120; kill $EVAL_PID 2>/dev/null; fi  # let the watcher finish the last checkpoint, then evaluate any it missed
  .venv/bin/python -m muscle_supermimic.tasks.tracking.scripts.evaluate --task $TASK --run-dir $RUN \
    --motion-file $TEST_MOTION --device cuda:0 >> logs/eval_$RUN_NAME.out 2>&1
  grep "^\[eval\]" logs/eval_$RUN_NAME.out | tail -5
  echo "== upload $RUN -> $HF_RESULTS_REPO/$RUN_NAME $(date -u +%FT%TZ)"
  INC=(--include "*.csv" --include "*.json" --include "params/*")  # hf upload takes one pattern per --include
  for P in $KEEP_CKPTS; do INC+=(--include "$P"); done
  .venv/bin/hf upload $HF_RESULTS_REPO $RUN $RUN_NAME --repo-type dataset --private "${INC[@]}" --quiet
  .venv/bin/hf upload $HF_RESULTS_REPO logs/train_$RUN_NAME.out $RUN_NAME/train_$RUN_NAME.out --repo-type dataset --quiet
  .venv/bin/hf upload $HF_RESULTS_REPO logs/eval_$RUN_NAME.out $RUN_NAME/eval_$RUN_NAME.out --repo-type dataset --quiet
fi
echo "== job done $(date -u +%FT%TZ)"
exit $TRAIN_STATUS
