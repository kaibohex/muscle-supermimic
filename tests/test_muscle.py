"""Analytic activation dynamics against MuJoCo stepping the full model."""

import mujoco
import numpy as np
import torch

from muscle_supermimic.assets.robots.mshuman700 import compiled_model
from muscle_supermimic.tasks.tracking.mdp.muscle import MuscleActivation


def test_activation_matches_mujoco():
    m = compiled_model()
    m.opt.timestep = 0.002
    substeps = 10
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[2] = 1.5  # in the air: contacts do not matter for the activation
    rng = np.random.default_rng(0)
    mus = MuscleActivation(m, substeps, m.opt.timestep, "cpu")
    act = torch.tensor(rng.uniform(0, 1, m.na))
    d.act[:] = act.numpy()
    for _ in range(4):
        ctrl = rng.uniform(0, 1, m.nu)
        d.ctrl[:] = ctrl
        mean_mj = np.zeros(m.na)
        for _ in range(substeps):
            mean_mj += d.act / substeps
            mujoco.mj_step(m, d)
        act, mean = mus.step(act.float(), torch.tensor(ctrl, dtype=torch.float32))
        np.testing.assert_allclose(act.numpy(), d.act, atol=1e-6)
        np.testing.assert_allclose(mean.numpy(), mean_mj, atol=1e-6)
        act = torch.tensor(d.act.copy())
