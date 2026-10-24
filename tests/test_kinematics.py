"""Torch FK (independent coordinates + coupled-joint polynomials) against MuJoCo's own kinematics."""

import mujoco
import numpy as np
import pytest
import torch

from muscle_supermimic.assets.robots.mshuman700 import ModelInfo, compiled_model
from muscle_supermimic.tasks.tracking.mdp.kinematics import MSHumanKinematics
from muscle_supermimic.utils.rotations import quat_to_mat, so3_exp

TRACKED = ("pelvis", "femur_r", "tibia_r", "calcn_r", "toes_r", "femur_l", "tibia_l", "calcn_l", "toes_l",
           "thoracic12", "head_neck", "humerus_r", "ulna_r", "hand_r", "humerus_l", "ulna_l", "hand_l")  # fmt: skip


def _random_states(info, n, rng):
    q = rng.uniform(info.q_lo, info.q_hi, size=(n, len(info.indep_ids)))
    qd = rng.normal(0, 2.0, size=q.shape)
    p = rng.normal(0, 1.0, size=(n, 3))
    R = so3_exp(torch.tensor(rng.normal(0, 1.5, size=(n, 3)))).numpy()
    v, w = rng.normal(0, 1.0, size=(n, 3)), rng.normal(0, 2.0, size=(n, 3))
    return p, R, v, w, q, qd


@pytest.mark.parametrize("lock_hands", [False, True])
def test_fk_matches_mujoco(lock_hands):
    m = compiled_model(lock_hands=lock_hands)
    info = ModelInfo(m)
    kin = MSHumanKinematics(TRACKED, "cpu", m, dtype=torch.float64)
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
        for k, name in enumerate(TRACKED):
            b = m.body(name).id
            np.testing.assert_allclose(bp[i, k].numpy(), d.xpos[b], atol=1e-9, err_msg=name)
            np.testing.assert_allclose(bR[i, k].numpy(), d.xmat[b].reshape(3, 3), atol=1e-9, err_msg=name)
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, b, vel, 0)
            np.testing.assert_allclose(bw[i, k].numpy(), vel[:3], atol=1e-8, err_msg=name)
            np.testing.assert_allclose(bv[i, k].numpy(), vel[3:], atol=1e-8, err_msg=name)


def test_coupled_joints_satisfy_equalities():
    m = compiled_model()
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


def test_canonical_root_is_identity_at_rest():
    m = compiled_model()
    kin = MSHumanKinematics(TRACKED, "cpu", m)
    R = kin.canonical_root(torch.tensor(m.body("pelvis").quat, dtype=torch.float32))
    torch.testing.assert_close(R, torch.eye(3), atol=1e-6, rtol=0)
    assert quat_to_mat(torch.tensor(m.body("pelvis").quat)).shape == (3, 3)
