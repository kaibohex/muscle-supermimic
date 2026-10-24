"""Torch muscle geometry vs MuJoCo: muscle lengths, velocities and forces at random poses (every robot variant)."""

import mujoco
import numpy as np
import pytest
import torch

from muscle_supermimic.assets.robots import mshuman700, myofullbody
from muscle_supermimic.assets.robots.model_info import ModelInfo
from muscle_supermimic.tasks.tracking.mdp.kinematics import SkeletonKinematics
from muscle_supermimic.tasks.tracking.mdp.muscle_geometry import MuscleGeometry

MODELS = {
    "mshuman700": lambda: mshuman700.compiled_model(),
    "mshuman700-locked-hands": lambda: mshuman700.compiled_model(lock_hands=True),
    "myofullbody": lambda: myofullbody.compiled_model(),
    "myofullbody-fingers": lambda: myofullbody.compiled_model(disable_fingers=False),
}


def _grazing(m, d, tol=1e-5):
    """Muscles with a site-geom-site piece whose straight segment passes within ``tol`` of the wrap surface
    (cylinders: in their plane). There the wrap switches on or off, and MuJoCo, MuJoCo-Warp and the torch
    code (which follows MuJoCo-Warp) differ by microns."""
    out = np.zeros(m.nu, dtype=bool)
    for a in range(m.nu):
        t = m.actuator_trnid[a, 0]
        for k in range(m.tendon_adr[t] + 1, m.tendon_adr[t] + m.tendon_num[t] - 1):
            if m.wrap_type[k] == mujoco.mjtWrap.mjWRAP_SITE:
                continue
            g = m.wrap_objid[k]
            R, c = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
            p0, p1 = (R.T @ (d.site_xpos[m.wrap_objid[j]] - c) for j in (k - 1, k + 1))
            if m.wrap_type[k] == mujoco.mjtWrap.mjWRAP_CYLINDER:
                p0, p1 = p0[:2], p1[:2]
            u = p1 - p0
            closest = p0 + np.clip(-p0 @ u / (u @ u), 0, 1) * u
            out[a] |= abs(np.linalg.norm(closest) - m.geom_size[g, 0]) < tol
    return out


@pytest.mark.parametrize("variant", MODELS)
def test_muscle_geometry_matches_mujoco(variant):
    m = MODELS[variant]()
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
        graze = _grazing(m, d)
        ok = ~graze
        np.testing.assert_allclose(L[i].numpy()[ok], d.actuator_length[ok], atol=1e-9, rtol=0)
        np.testing.assert_allclose(Ld[i].numpy()[ok], d.actuator_velocity[ok], atol=1e-8, rtol=0)
        np.testing.assert_allclose(F[i].numpy()[ok], d.actuator_force[ok], atol=1e-6, rtol=1e-8)
        np.testing.assert_allclose(L[i].numpy()[graze], d.actuator_length[graze], atol=1e-5, rtol=0)
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
