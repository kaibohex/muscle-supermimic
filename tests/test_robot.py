"""The muscle-to-joint map behind ``lock_hands``, and the locked model it produces."""

import mujoco
import numpy as np

from muscle_supermimic.assets.robots.mshuman700 import ModelInfo, compiled_model
from muscle_supermimic.assets.robots.mshuman700.mshuman700_cfg import muscle_spans

WRIST = {"deviation_r", "flexion_r", "deviation_l", "flexion_l"}


def _moment_arms(m, info, q, delta=1e-4):
    """|dL/dq| of every tendon w.r.t. every independent joint (coupled joints follow), (ntendon, 37)."""
    d = mujoco.MjData(m)

    def lengths(x):
        d.qpos[:] = 0
        d.qpos[3] = 1.0
        d.qpos[7:] = info.full_joint_pos(x[None])[0][0]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_tendon(m, d)
        return d.ten_length.copy()

    out = np.zeros((m.ntendon, len(q)))
    for i in range(len(q)):
        qp, qm = q.copy(), q.copy()
        qp[i] += delta
        qm[i] -= delta
        out[:, i] = np.abs(lengths(qp) - lengths(qm)) / (2 * delta)
    return out


def test_muscle_spans_match_moment_arms():
    m = compiled_model()
    info = ModelInfo(m)
    spans = muscle_spans(m, info)
    rng = np.random.default_rng(0)
    arms = np.max([_moment_arms(m, info, rng.uniform(info.q_lo, info.q_hi)) for _ in range(3)], 0)
    for a, name in enumerate(info.muscle_names):
        acts = {info.indep_names[i] for i in np.nonzero(arms[m.actuator_trnid[a, 0]] > 1e-6)[0]}
        assert acts == spans[name], (name, acts ^ spans[name])


def test_lock_hands_model():
    m0, m1 = compiled_model(), compiled_model(lock_hands=True)
    i0, i1 = ModelInfo(m0), ModelInfo(m1)
    spans = muscle_spans(m0, i0)
    assert set(i0.indep_names) - set(i1.indep_names) == WRIST
    assert not any(n.startswith("wrist_hand") for n in i1.joint_names)
    assert len(i1.coupled) == len(i0.coupled) - 4
    # exactly the muscles acting on nothing but the wrist are gone; the others keep their tendons
    removed = set(i0.muscle_names) - set(i1.muscle_names)
    assert removed == {a for a, s in spans.items() if s and s <= WRIST}
    assert dict(zip(i1.muscle_names, i1.muscle_tendon_names)) == {
        a: t for a, t in zip(i0.muscle_names, i0.muscle_tendon_names) if a not in removed
    }
    assert np.isclose(m0.body_mass.sum(), m1.body_mass.sum())
    # the kept wrist-crossing muscles now act only on the remaining joints
    spans1 = muscle_spans(m1, i1)
    for a, s in spans.items():
        if a not in removed:
            assert spans1[a] == s - WRIST, a
