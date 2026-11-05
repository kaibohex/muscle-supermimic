"""Batched checkpoint evaluation (``ms-eval --batch K``): K policies at once in one env of K x C worlds.

World k C + j plays clip j with policy k. The same policy in two groups gives the same rollout, and each group's result
is its own policy's. Needs ``data/myofullbody/walk.npz``.
"""

import copy
import os

import pytest
import torch

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
TASK, CLIP = "Mjlab-SuperMimic-Flat-MyoFullBody-No-Muscle-Obs", os.path.join(DATA, "myofullbody", "walk.npz")


def test_batched_evaluation_keeps_each_policy_to_its_own_worlds():
    from muscle_supermimic.tasks.tracking.scripts.evaluate import build, evaluate_many

    if not os.path.exists(CLIP):
        pytest.skip(f"no test clip at {CLIP}")
    torch.manual_seed(0)
    env, runner = build(TASK, CLIP, "cpu", copies=3)
    assert env.unwrapped.num_envs == 3 * len(runner.cmd.motion.names)
    p0 = copy.deepcopy(runner.deployed).eval()
    p1 = copy.deepcopy(p0)
    with torch.no_grad():
        for w in p1.parameters():
            if w.dim() == 2:
                w.add_(0.3 * torch.randn_like(w) * w.std())
    r0, r1, r2 = evaluate_many(env, runner, [p0, p1, p0])
    # the same policy, the same rollout up to float rounding (the worlds share MuJoCo-Warp's contact pools, so what the
    # other groups do moves the last bits: 1e-8 here, ~1e-3 of a clip's return over a long closed-loop rollout); another
    # policy in the middle group (an untrained policy fails within a few steps either way: compare unrounded returns)
    assert abs(r0["norm_return"] - r2["norm_return"]) < 1e-5 and r0["n_failed"] == r2["n_failed"]
    assert r0["norm_return"] != r1["norm_return"]
    # fewer policies than copies: the spare worlds follow the last policy and are left out of the results
    (only,) = evaluate_many(env, runner, [p1])
    assert abs(only["norm_return"] - r1["norm_return"]) < 1e-3
