"""MyoFullBody as MuscleMimic configures it, and its retargeted clips read through the motion library."""

import os

import mujoco
import numpy as np
import pytest
import torch

from muscle_supermimic.assets.robots.model_info import ModelInfo, root_offset
from muscle_supermimic.assets.robots.myofullbody import MYOFULLBODY_XML, compiled_model
from muscle_supermimic.assets.robots.myofullbody.myofullbody_cfg import get_spec
from muscle_supermimic.assets.robots.myofullbody.myofullbody_constants import FINGER_JOINTS, FINGER_MUSCLES
from muscle_supermimic.tasks.tracking.config.myofullbody.env_cfgs import TRACKED_BODIES
from muscle_supermimic.tasks.tracking.mdp.kinematics import SkeletonKinematics
from muscle_supermimic.tasks.tracking.mdp.motion import load_clip
from muscle_supermimic.utils.rotations import quat_to_mat

CLIP = os.path.join(os.path.dirname(__file__), "..", "data", "myofullbody", "walk.npz")


def test_disable_fingers_matches_musclemimic():
    up = mujoco.MjSpec.from_file(str(MYOFULLBODY_XML))
    # MuscleMimic deletes the tendons whose name contains a finger muscle's name
    up_tendons = {t.name for t in up.tendons if any(f in t.name for f in FINGER_MUSCLES)}
    m0, m1 = compiled_model(disable_fingers=False), compiled_model()
    i0, i1 = ModelInfo(m0), ModelInfo(m1)
    assert set(i0.joint_names) - set(i1.joint_names) == set(FINGER_JOINTS)
    assert set(i0.muscle_names) - set(i1.muscle_names) == set(FINGER_MUSCLES)
    names = lambda m: {m.tendon(t).name for t in range(m.ntendon)}  # noqa: E731
    assert names(m0) - names(m1) == up_tendons
    # MuscleMimic's counts: 83 joints (root included), 354 muscles; the 51 equalities all remain
    assert (m1.njnt, m1.nu, m1.neq, len(i1.coupled), len(i1.indep_ids)) == (83, 354, 51, 51, 31)
    assert np.isclose(m0.body_mass.sum(), m1.body_mass.sum())


def test_spec_changes():
    m = compiled_model()
    assert m.nkey == 0 and m.nlight == 0 and not any(m.geom_bodyid == 0)  # no scene left
    assert (m.sensor_type == mujoco.mjtSensor.mjSENS_TOUCH).all() and m.nsensor == 4
    assert m.npair == 75 and get_spec(self_collisions=False).compile().npair == 0
    # robot geoms collide with the ground only (contype 1, conaffinity 0), self-contacts only through pairs
    hit = m.geom_contype > 0
    assert hit.any() and not m.geom_conaffinity[hit].any()
    np.testing.assert_allclose(root_offset(m), [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], atol=1e-6)


@pytest.mark.skipif(not os.path.exists(CLIP), reason="no MuscleMimic clip at data/myofullbody/walk.npz (see README)")
def test_clip_matches_musclemimic_bodies():
    """FK of a loaded retargeted clip reproduces the body poses MuscleMimic stored with it, except below the
    left knee: musclemimic-models <= 1.0.5 coupled knee_angle_rotation3_l with the negated polynomial and
    the retargeted caches were made with it; 1.0.6 mirrors the right knee. The motion library recomputes
    coupled joints from the independent ones with the installed model's equalities."""
    m = compiled_model()
    info = ModelInfo(m)
    kin = SkeletonKinematics(TRACKED_BODIES, "cpu", m, dtype=torch.float64)
    fps, p, quat, q = load_clip(CLIP, info, root_offset(m))
    data = np.load(CLIP)
    assert fps == float(data["frequency"]) and len(p) == len(data["qpos"])
    t = lambda x: torch.tensor(x, dtype=torch.float64)  # noqa: E731
    zero = torch.zeros(len(p), 3, dtype=torch.float64)
    bp, bR, _, _ = kin.forward(t(p), quat_to_mat(t(quat)), zero, zero, t(q), torch.zeros_like(t(q)))

    names = [str(n) for n in data["joint_names"]]
    col = info.joint_names.index("knee_angle_rotation3_l")
    clip_r3 = data["qpos"][:, 7 + names.index("knee_angle_rotation3_l") - 1]
    np.testing.assert_allclose(clip_r3, -info.full_joint_pos(q)[0][:, col], atol=2e-3)
    bodies = [str(n) for n in data["body_names"]]
    for k, name in enumerate(TRACKED_BODIES):
        if name in ("tibia_l", "talus_l", "toes_l"):
            continue
        b = bodies.index(name)
        np.testing.assert_allclose(bp[:, k].numpy(), data["xpos"][:, b], atol=1e-3, err_msg=name)
        R_ref = quat_to_mat(t(data["xquat"][:, b].astype(np.float64))).numpy()
        np.testing.assert_allclose(bR[:, k].numpy(), R_ref, atol=2e-3, err_msg=name)
