"""Torch FK (independent coordinates + coupled-joint polynomials) against MuJoCo's own kinematics."""

import mujoco
import numpy as np
import pytest
import torch

from muscle_supermimic.assets.robots import mshuman700
from muscle_supermimic.assets.robots.model_info import ModelInfo
from muscle_supermimic.tasks.tracking.config.mshuman700.env_cfgs import TRACKED_BODIES as MSHUMAN700_TRACKED
from muscle_supermimic.tasks.tracking.mdp.kinematics import SkeletonKinematics
from muscle_supermimic.utils.rotations import so3_exp

# (compiled entity model, tracked bodies) of every robot variant
MODELS = {
    "mshuman700": (lambda: mshuman700.compiled_model(), MSHUMAN700_TRACKED),
    "mshuman700-locked-hands": (lambda: mshuman700.compiled_model(lock_hands=True), MSHUMAN700_TRACKED),
}


def _random_states(info, n, rng):
    q = rng.uniform(info.q_lo, info.q_hi, size=(n, len(info.indep_ids)))
    qd = rng.normal(0, 2.0, size=q.shape)
    p = rng.normal(0, 1.0, size=(n, 3))
    R = so3_exp(torch.tensor(rng.normal(0, 1.5, size=(n, 3)))).numpy()
    v, w = rng.normal(0, 1.0, size=(n, 3)), rng.normal(0, 2.0, size=(n, 3))
    return p, R, v, w, q, qd


@pytest.mark.parametrize("variant", MODELS)
def test_fk_matches_mujoco(variant):
    model_fn, tracked = MODELS[variant]
    m = model_fn()
    info = ModelInfo(m)
    kin = SkeletonKinematics(tracked, "cpu", m, dtype=torch.float64)
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    n = 16
    p, R, v, w, q, qd = _random_states(info, n, rng)
    t = lambda x: torch.tensor(x, dtype=torch.float64)  # noqa: E731
    bp, bR, bv, bw = kin.forward(t(p), t(R), t(v), t(w), t(q), t(qd))
    qf, qdf = info.full_joint_pos(q, qd)
    Rb = (t(R) @ kin.R_off.T).numpy()
    for i in range(n):
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, Rb[i].reshape(-1))
        d.qpos[:7] = np.concatenate((p[i], quat))
        d.qvel[:3], d.qvel[3:6] = v[i], Rb[i].T @ w[i]
        d.qpos[7:], d.qvel[6:] = qf[i], qdf[i]
        mujoco.mj_forward(m, d)
        for k, name in enumerate(tracked):
            b = m.body(name).id
            np.testing.assert_allclose(bp[i, k].numpy(), d.xpos[b], atol=1e-9, err_msg=name)
            np.testing.assert_allclose(bR[i, k].numpy(), d.xmat[b].reshape(3, 3), atol=1e-9, err_msg=name)
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, b, vel, 0)
            np.testing.assert_allclose(bw[i, k].numpy(), vel[:3], atol=1e-8, err_msg=name)
            np.testing.assert_allclose(bv[i, k].numpy(), vel[3:], atol=1e-8, err_msg=name)


@pytest.mark.parametrize("variant", MODELS)
def test_coupled_joints_satisfy_equalities(variant):
    m = MODELS[variant][0]()
    info = ModelInfo(m)
    d = mujoco.MjData(m)
    rng = np.random.default_rng(1)
    q = rng.uniform(info.q_lo, info.q_hi, size=(1, len(info.indep_ids)))
    d.qpos[3] = 1.0
    d.qpos[7:] = info.full_joint_pos(q)[0][0]
    mujoco.mj_forward(m, d)
    eq_rows = d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    assert eq_rows.sum() == len(info.coupled)
    np.testing.assert_allclose(d.efc_pos[eq_rows], 0.0, atol=1e-12)


@pytest.mark.parametrize("variant", MODELS)
def test_canonical_root_is_identity_at_rest(variant):
    model_fn, tracked = MODELS[variant]
    m = model_fn()
    kin = SkeletonKinematics(tracked, "cpu", m)
    R = kin.canonical_root(torch.tensor(m.qpos0[3:7], dtype=torch.float32))
    torch.testing.assert_close(R, torch.eye(3), atol=1e-6, rtol=0)
    # and that default pose stands upright facing +x: toes ahead of heels, left hip at +y
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    x = lambda name: d.xpos[m.body(name).id]  # noqa: E731
    assert x("toes_r")[0] > x("calcn_r")[0] + 0.1 and x("femur_l")[1] > x("femur_r")[1] + 0.1
    assert x(tracked[0])[2] > x("toes_r")[2] + 0.7
