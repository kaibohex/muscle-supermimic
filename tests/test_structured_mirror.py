"""BlinkMimic's structured (semi-implicit) world model and the left-right mirror maps of the muscle models."""

import pytest
import torch

from muscle_supermimic.rl.supermimic.networks import WorldModel
from muscle_supermimic.utils.rotations import quat_to_mat


def _state(B: int, nq: int, gen: torch.Generator):
    rn = lambda *s: torch.randn(*s, generator=gen)  # noqa: E731
    return (rn(B, 3), quat_to_mat(rn(B, 4)), rn(B, 3), rn(B, 3), 0.5 * rn(B, nq), rn(B, nq))


@pytest.mark.parametrize("structured", [False, True])
def test_integrate_inverts_targets(structured):
    gen = torch.Generator().manual_seed(0)
    nq = 7
    wm = WorldModel(nq, 5, 4, hidden=(16,), structured=structured, dt=0.01, substeps=5)
    S, S1 = _state(8, nq, gen), _state(8, nq, gen)
    t = wm.targets(S, S1)
    S1_hat = wm.integrate(S, t)
    for a, b in zip(S1_hat, S1, strict=True):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    assert WorldModel.state_error(S1_hat, S1).abs().max() < 1e-5
    inc = WorldModel.increments(S, S1)
    if structured:  # the velocity entries are the increments; the position entries are corrections on the drift
        torch.testing.assert_close(t[:, 6:12], inc[:, 6:12], atol=0, rtol=0)
        torch.testing.assert_close(t[:, 12 + nq :], inc[:, 12 + nq :], atol=0, rtol=0)
        assert not torch.allclose(t[:, :3], inc[:, :3]) and not torch.allclose(t[:, 12 : 12 + nq], inc[:, 12 : 12 + nq])
    else:
        torch.testing.assert_close(t, inc, atol=0, rtol=0)


def test_drift_is_the_semi_implicit_euler_position_update():
    """n substeps p += h v_k with v linear from v0 to vn give dt ((1 - W) v0 + W vn), W = (n + 1) / (2 n)."""
    gen = torch.Generator().manual_seed(1)
    n, dt = 5, 0.01
    v0, vn, w0, wn = (torch.randn(3, generator=gen) for _ in range(4))
    qd0, qdn = (torch.randn(2, generator=gen) for _ in range(2))
    p = torch.zeros(3)
    for k in range(1, n + 1):
        p = p + dt / n * (v0 + (vn - v0) * k / n)
    W = (n + 1) / (2 * n)
    torch.testing.assert_close(p, dt * ((1 - W) * v0 + W * vn))
    wm = WorldModel(2, 1, 4, hidden=(8,), structured=True, dt=dt, substeps=n)
    assert wm.drift_w == W
    S = (torch.zeros(1, 3), torch.eye(3)[None], v0[None], w0[None], torch.zeros(1, 2), qd0[None])
    gp, gth, gq = wm._drift(S, vn[None], wn[None], qdn[None])  # identity root frame: the drift is the formula itself
    torch.testing.assert_close(gp[0], p)
    torch.testing.assert_close(gth[0], dt * ((1 - W) * w0 + W * wn))
    torch.testing.assert_close(gq[0], dt * ((1 - W) * qd0 + W * qdn))


# ---------------------------------------------------------------- mirror maps of the two models


def _models():
    from muscle_supermimic.assets.robots.mshuman700 import compiled_model as ms_model
    from muscle_supermimic.assets.robots.myofullbody import compiled_model as myo_model
    from muscle_supermimic.tasks.tracking.config.mshuman700.env_cfgs import FOOT_BODIES as MS_FEET
    from muscle_supermimic.tasks.tracking.config.mshuman700.env_cfgs import TRACKED_BODIES as MS_TRACKED
    from muscle_supermimic.tasks.tracking.config.myofullbody.env_cfgs import FOOT_BODIES as MYO_FEET
    from muscle_supermimic.tasks.tracking.config.myofullbody.env_cfgs import TRACKED_BODIES as MYO_TRACKED

    return {
        "myofullbody": (lambda: myo_model(), MYO_TRACKED, MYO_FEET),
        "mshuman700_locked_hands": (lambda: ms_model(lock_hands=True), MS_TRACKED, MS_FEET),
    }


@pytest.fixture(scope="module", params=["myofullbody", "mshuman700_locked_hands"])
def mirror(request):
    from muscle_supermimic.tasks.tracking.mdp.kinematics import SkeletonKinematics
    from muscle_supermimic.tasks.tracking.mdp.mirror import Mirror
    from muscle_supermimic.tasks.tracking.mdp.muscle_geometry import MuscleGeometry

    make, tracked, feet = _models()[request.param]
    m = make()
    kin, geo = SkeletonKinematics(tracked, "cpu", m), MuscleGeometry(m, "cpu")
    return Mirror(kin, geo, feet, "cpu"), kin, geo


def test_mirror_maps_are_symmetric_involutions(mirror):
    from muscle_supermimic.tasks.tracking.mdp.mirror import partner_name

    mr, kin, geo = mirror
    # the constructor validated the maps against the model's own asymmetry (docstring of mirror.py): MS-Human-700's
    # skeleton is exact, MyoFullBody's symmetric to 2 mm; muscle paths to < 1 mm but for a few known muscles
    r = mr.residuals
    assert r["positions"] < 3e-3 and r["rotations"] < 1e-2 and r["default_pose"] < 1e-3, r
    assert int((mr.muscle_residual > 1e-3).sum()) <= 8 and int((mr.muscle_residual > 5e-3).sum()) <= 6, r
    assert float(mr.muscle_residual.median()) < 1e-5 and float(mr.muscle_residual.max()) < 5e-2
    names = kin.info.indep_names
    for k, name in enumerate(names):  # the joint map found from the kinematics agrees with the names
        assert names[int(mr.j_src[k])] == partner_name(name), (name, names[int(mr.j_src[k])])
    mus = kin.info.muscle_names
    n_side = 0
    for i, name in enumerate(mus):  # muscles pair by name; a midline muscle maps to itself
        p = mus[int(mr.muscle_perm[i])]
        if p != name:
            n_side += 1
            assert p in (partner_name(name), name + "_left", name + "_L", name + "_l") or name in (p + "_left", p + "_L", p + "_l"), (name, p)
    assert n_side == len(mus)  # no midline muscle in either model
    assert mr.touch_perm.tolist() == [2, 3, 0, 1]  # right heel, right toes, left heel, left toes
    # involutions
    gen = torch.Generator().manual_seed(0)
    S = _state(4, kin.num_q, gen)
    for a, b in zip(mr.body(mr.body(S)), S, strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    x = torch.randn(4, geo.num_muscles, generator=gen)
    torch.testing.assert_close(mr.muscles(mr.muscles(x)), x, atol=0, rtol=0)
    t = torch.randn(4, 4, generator=gen)
    torch.testing.assert_close(mr.touch(mr.touch(t)), t, atol=0, rtol=0)
    # a left-side joint motion becomes the same right-side motion; the spine's lateral bending and axial rotation flip
    q0 = mr.q0.clone()
    for name, sign in (("hip_flexion_l", 1.0), ("knee_angle_l", 1.0), ("elbow_flex_l", 1.0), ("elbow_flexion_l", 1.0),
                       ("lat_bending", -1.0), ("axial_rotation", -1.0), ("flex_extension", 1.0),
                       ("L5_S1_LB", -1.0), ("L5_S1_AR", -1.0), ("L5_S1_FE", 1.0)):  # fmt: skip
        if name not in names:
            continue
        q = q0.clone()
        q[names.index(name)] += 0.3
        qm = mr.joints(q)
        assert torch.isclose(qm[names.index(partner_name(name))] - q0[names.index(partner_name(name))], torch.tensor(sign * 0.3)), name
