"""Torch muscle geometry vs MuJoCo: muscle lengths, velocities and forces at random poses (both models)."""

import mujoco
import numpy as np
import pytest
import torch

from muscle_supermimic.assets.robots.model_info import ModelInfo
from muscle_supermimic.assets.robots.mshuman700 import compiled_model
from muscle_supermimic.tasks.tracking.mdp.kinematics import SkeletonKinematics
from muscle_supermimic.tasks.tracking.mdp.muscle_geometry import MuscleGeometry


@pytest.mark.parametrize("lock_hands", [False, True])
def test_muscle_geometry_matches_mujoco(lock_hands):
    m = compiled_model(lock_hands=lock_hands)
    info = ModelInfo(m)
    geo = MuscleGeometry(m, "cpu", torch.float64)
    kin = SkeletonKinematics(("pelvis",), "cpu", m, torch.float64)
    rng = np.random.default_rng(0)
    d = mujoco.MjData(m)
    n, wrapped = 16, 0
    q = torch.tensor(rng.uniform(info.q_lo, info.q_hi, (n, len(info.indep_ids))))
    qd = torch.tensor(rng.normal(0, 2.0, q.shape))
    act = torch.tensor(rng.uniform(0, 1, (n, m.nu)))
    L, Ld = geo.length_velocity(q, qd)
    F = geo.force(L, Ld, act)
    qf, qdf = kin.full_joint_state(q, qd)
    for i in range(n):
        d.qpos[:] = 0
        d.qpos[3] = 1.0
        d.qpos[7:] = qf[i].numpy()
        d.qvel[:] = 0
        d.qvel[6:] = qdf[i].numpy()
        d.act[:] = act[i].numpy()
        mujoco.mj_forward(m, d)
        np.testing.assert_allclose(L[i].numpy(), d.actuator_length, atol=1e-9, rtol=0)
        np.testing.assert_allclose(Ld[i].numpy(), d.actuator_velocity, atol=1e-8, rtol=0)
        np.testing.assert_allclose(F[i].numpy(), d.actuator_force, atol=1e-6, rtol=1e-8)
        wrapped += int((d.wrap_xpos.reshape(-1, 6)[:, 3:].any(-1)).sum())
    assert wrapped > 0  # the poses exercise the wraps
    # gradients exist and are finite
    qg = q.clone().requires_grad_()
    Lg, Ldg = geo.length_velocity(qg, qd)
    (geo.force(Lg, Ldg, act).sum() + Lg.sum()).backward()
    assert torch.isfinite(qg.grad).all() and qg.grad.abs().sum() > 0
    # length gradient == finite differences along random directions
    for _ in range(3):
        dq = torch.tensor(rng.normal(0, 1, q.shape))
        h = 1e-6
        fd = (geo.length_velocity(q + h * dq, qd)[0] - geo.length_velocity(q - h * dq, qd)[0]) / (2 * h)
        qg = q.clone().requires_grad_()
        jvp = torch.autograd.grad(geo.length_velocity(qg, qd)[0], qg, torch.ones(n, m.nu, dtype=torch.float64))[0]
        torch.testing.assert_close((jvp * dq).sum(), fd.sum(), atol=1e-5, rtol=1e-6)
