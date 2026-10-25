"""MyoFullBody -> MS-Human-700 retargeting: what carries over is exact, what is fitted is reached, and a
retargeted clip is a motion library clip whose feet touch the ground as the source's do."""

import os

import numpy as np
import pytest
import torch

from muscle_supermimic.assets.robots import mshuman700
from muscle_supermimic.assets.robots.model_info import ModelInfo, root_offset
from muscle_supermimic.tasks.tracking.config.mshuman700.env_cfgs import TRACKED_BODIES
from muscle_supermimic.tasks.tracking.mdp.kinematics import SkeletonKinematics
from muscle_supermimic.tasks.tracking.mdp.motion import MotionLib, load_clip
from muscle_supermimic.tasks.tracking.scripts.retarget_myofullbody import (
    FITTED,
    MATCHED,
    Retargeter,
    lowest_foot_point,
)

CLIP = os.path.join(os.path.dirname(__file__), "..", "data", "myofullbody", "walk.npz")
LEGS = ("femur", "tibia", "talus", "calcn", "toes")
ARMS = ("humerus", "ulna", "radius")


@pytest.fixture(scope="module")
def rt():
    return Retargeter()


def _random_source(rt, n, rng, spread=1.0):
    """MyoFullBody poses with every joint uniform in its range shrunk by ``spread`` around 0."""
    info = rt.mf_info
    lo, hi = np.minimum(info.q_lo, 0) * spread, np.maximum(info.q_hi, 0) * spread
    return rng.uniform(lo, hi, (n, len(info.indep_names)))


def test_transferred_joints_turn_segments_alike(rt):
    """Each leg and arm segment turns away from its rest orientation alike in both models (in its own frame),
    for angles inside both models' ranges: the same joints about the same axes."""
    bodies = [f"{b}_{s}" for b in LEGS + ARMS for s in "rl"]
    q_mf = _random_source(rt, 64, np.random.default_rng(0))
    src, dst = rt.src.numpy(), rt.dst.numpy()
    lo = np.maximum(rt.mf_info.q_lo[src], rt.ms_info.q_lo[dst])
    hi = np.minimum(rt.mf_info.q_hi[src], rt.ms_info.q_hi[dst])
    q_mf[:, src] = np.random.default_rng(1).uniform(lo, hi, (len(q_mf), len(src)))
    for n in ("flex_extension", "lat_bending", "axial_rotation"):  # thorax at rest, so the spine fit is exact
        q_mf[:, rt.mf_info.indep_names.index(n)] = 0.0
    q_ms, _ = rt(q_mf)

    def local_turn(m, q):
        kin = SkeletonKinematics(bodies, "cpu", m, dtype=torch.float64)
        n = len(q)
        z, eye = torch.zeros(n, 3, dtype=torch.float64), torch.eye(3, dtype=torch.float64).expand(n, 3, 3)
        R = kin.forward(z, eye, z, z, torch.tensor(q), torch.zeros(n, kin.num_q, dtype=torch.float64))[1]
        R0 = kin.forward(z[:1], eye[:1], z[:1], z[:1], *(torch.zeros(1, kin.num_q, dtype=torch.float64),) * 2)[1]
        return R0.transpose(-1, -2) @ R  # world rotation is the root's (identity): segment's turn in its own rest frame

    # 1e-5: the arms hang off the thorax, which the spine fit holds at rest to ~1e-6 rad
    torch.testing.assert_close(local_turn(rt.ms, q_ms), local_turn(rt.mf, q_mf), atol=1e-5, rtol=0)


def test_fit_reaches_thorax_and_hand_flexion(rt):
    """Inside what MS-Human-700's spine and wrist flexion can reach (and with no wrist deviation, which stays
    at 0), the fit reproduces the thorax's rotation, and the hands' up to the ~2 deg between the two models'
    flexion axes."""
    q_mf = _random_source(rt, 32, np.random.default_rng(1), spread=0.25)
    for n in ("deviation_r", "deviation_l"):
        q_mf[:, rt.mf_info.indep_names.index(n)] = 0.0
    q_ms, stats = rt(q_mf)
    for (seg, _), _ in MATCHED:
        assert stats[f"{seg}_deg"].max() < (0.5 if seg == "thoracic12" else 1.0), seg
    fit = q_ms[:, [rt.ms_info.indep_names.index(n) for n in FITTED]]
    lo, hi = rt.lo[rt.fit].numpy(), rt.hi[rt.fit].numpy()
    assert ((fit > lo + 1e-3) & (fit < hi - 1e-3)).all()  # an interior solution, not a limit


@pytest.mark.skipif(not os.path.exists(CLIP), reason="no MuscleMimic clip at data/myofullbody/walk.npz (see README)")
@pytest.mark.parametrize("lock_hands", [False, True])
def test_retargeted_clip(rt, tmp_path, lock_hands):
    fps, p_mf, quat, q_mf = load_clip(CLIP, rt.mf_info, rt.mf_R_off)
    p, q, stats = rt.clip(p_mf, quat, q_mf)
    assert stats["thoracic12_deg"].max() < 0.5 and stats["clamped"] < 0.02
    out = tmp_path / "walk.npz"
    np.savez(out, fps=fps, root_pos=p, root_quat=quat, joint_pos=q, joint_names=np.array(rt.ms_info.indep_names))
    m = mshuman700.compiled_model(lock_hands=lock_hands)
    lib = MotionLib([str(out)], SkeletonKinematics(TRACKED_BODIES, "cpu", m), 0.01, "cpu")
    assert lib.num_frames == len(q) and lib.q.shape[-1] == len(ModelInfo(m).indep_names)
    # the feet touch the ground as in the source
    _, p_ms, quat_ms, q_ms = load_clip(str(out), rt.ms_info, root_offset(rt.ms))
    dz = lowest_foot_point(rt.ms, p_ms, quat_ms, q_ms) - lowest_foot_point(rt.mf, p_mf, quat, q_mf)
    assert abs(np.median(dz)) < 1e-3 and np.quantile(np.abs(dz), 0.99) < 5e-3
