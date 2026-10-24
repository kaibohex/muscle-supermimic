"""``EntityCfg`` for the free-floating MS-Human-700, plus the joint/muscle bookkeeping the task needs.

The upstream file is loaded through ``MjSpec`` and changed in these ways:

1. the six pelvis DoF become one free joint, so the root has no gimbal lock and its state is the
   (p, R, v, w) tuple SuperMimic works with (mjlab also requires a free joint for a floating base);
2. the standalone-scene decoration (floor, lights, cameras, keyframe, ``<size>`` hints) goes, since
   mjlab's ``SceneCfg`` owns the ground and the solver budget;
3. optionally (default), robot-robot contacts are disabled while robot-ground contacts stay, as in
   SuperMimic;
4. optionally (``lock_hands``), the hand + wrist DoF are deleted (each hand welded to its radius in
   the neutral pose) together with their couplers and the muscles that act on nothing else.

Otherwise muscles, wrapping tendons, ``<equality>`` couplers, joint damping/stiffness are upstream.
"""

from __future__ import annotations

import functools

import mujoco
import numpy as np
from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from muscle_supermimic.assets.robots.mshuman700.mshuman700_constants import (
    FREE_JOINT,
    INIT_POS,
    INIT_ROT,
    MUSCLE_TENDON_EXPR,
    PELVIS_JOINTS,
    ROOT_BODY,
    WRIST_ROOT_BODIES,
    require_mshuman700_xml,
)


def _walk(body: mujoco.MjsBody):
    yield body
    for child in body.bodies:
        yield from _walk(child)


def _strip_scene(spec: mujoco.MjSpec) -> None:
    """Remove what mjlab's scene provides itself (same surgery as muscle-mjlab's MS-Human-700 config)."""
    spec.njmax = -1
    spec.nconmax = -1
    spec.nkey = 0
    spec.nuser_jnt = -1
    spec.visual.map.znear = mujoco.MjSpec().visual.map.znear
    for key in list(spec.keys):
        spec.delete(key)
    for geom in list(spec.worldbody.geoms):  # the floor plane
        spec.delete(geom)
    for body in _walk(spec.worldbody):
        for light in list(body.lights):
            spec.delete(light)
        for camera in list(body.cameras):
            spec.delete(camera)
    for texture in list(spec.textures):
        spec.delete(texture)
    for material in list(spec.materials):
        spec.delete(material)


def get_spec(self_collisions: bool = False, lock_hands: bool = False) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(require_mshuman700_xml()))
    _strip_scene(spec)
    pelvis = spec.body(ROOT_BODY)
    for joint in list(pelvis.joints):
        assert joint.name in PELVIS_JOINTS, joint.name
        spec.delete(joint)
    pelvis.add_freejoint(name=FREE_JOINT)
    if not self_collisions:
        # robot geoms keep contype=1 (they hit the ground plane) but get conaffinity=0 (never each other)
        for body in _walk(pelvis):
            for geom in body.geoms:
                if geom.contype or geom.conaffinity:
                    geom.contype, geom.conaffinity = 1, 0
    if lock_hands:
        _lock_hands(spec)
    return spec


def _lock_hands(spec: mujoco.MjSpec) -> None:
    """Delete every joint in the WRIST_ROOT_BODIES subtrees (welding the hands to the forearms at q = 0,
    the neutral wrist, which satisfies the wrist couplers), the couplers on those joints, and every muscle
    whose tendon crosses no other joint. Muscles crossing the wrist *and* the elbow or the radioulnar
    joint stay. Order: actuators before their tendons, couplers before their joints."""
    m = compiled_model()
    info = ModelInfo(m)
    locked = set(_subtree_joints(m, WRIST_ROOT_BODIES))
    spans = muscle_spans(m, info)
    doomed = {a for a, s in spans.items() if s and s <= locked}
    doomed_tendons = {t for a, t in zip(info.muscle_names, info.muscle_tendon_names, strict=True) if a in doomed}
    kept_sites = set().union(*(_tendon_sites(m, t) for t in range(m.ntendon) if m.tendon(t).name not in doomed_tendons))
    doomed_sites = set().union(*(_tendon_sites(m, m.tendon(t).id) for t in doomed_tendons)) - kept_sites
    for actuator in list(spec.actuators):
        if actuator.name in doomed:
            spec.delete(actuator)
    for tendon in list(spec.tendons):
        if tendon.name in doomed_tendons:
            spec.delete(tendon)
    for site in list(spec.sites):
        if site.name in doomed_sites:
            spec.delete(site)
    for equality in list(spec.equalities):
        if equality.name1 in locked or equality.name2 in locked:
            spec.delete(equality)
    for joint in list(spec.joints):
        if joint.name in locked:
            spec.delete(joint)


def _subtree_joints(m: mujoco.MjModel, roots) -> list[str]:
    root_ids = {m.body(r).id for r in roots}
    out = []
    for j in range(m.njnt):
        b = int(m.jnt_bodyid[j])
        while b != 0 and b not in root_ids:
            b = int(m.body_parentid[b])
        if b in root_ids:
            out.append(m.joint(j).name)
    return out


def _tendon_sites(m: mujoco.MjModel, t: int) -> set[str]:
    """Names of the sites a tendon's path uses: via points and wrap side sites."""
    out = set()
    for k in range(m.tendon_adr[t], m.tendon_adr[t] + m.tendon_num[t]):
        if m.wrap_type[k] == mujoco.mjtWrap.mjWRAP_SITE:
            out.add(m.site(int(m.wrap_objid[k])).name)
        elif m.wrap_type[k] in (mujoco.mjtWrap.mjWRAP_SPHERE, mujoco.mjtWrap.mjWRAP_CYLINDER) and m.wrap_prm[k] >= 0:
            out.add(m.site(int(m.wrap_prm[k])).name)
    return out


def muscle_spans(m: mujoco.MjModel, info: ModelInfo) -> dict[str, set[str]]:
    """Independent joints each muscle acts on: the joints its tendon path crosses (path points and wrap
    surfaces on both sides of the joint), a coupled joint counting as the independent joint driving it.
    Exact and threshold-free (tests check it against finite-difference moment arms)."""
    def ancestors(b):
        out = {b}
        while b != 0:
            b = int(m.body_parentid[b])
            out.add(b)
        return out

    anc = [ancestors(b) for b in range(m.nbody)]
    spans = {}
    for a in range(m.nu):
        t = m.actuator_trnid[a, 0]
        bodies = set()
        for k in range(m.tendon_adr[t], m.tendon_adr[t] + m.tendon_num[t]):
            if m.wrap_type[k] == mujoco.mjtWrap.mjWRAP_SITE:
                bodies.add(int(m.site_bodyid[m.wrap_objid[k]]))
            elif m.wrap_type[k] in (mujoco.mjtWrap.mjWRAP_SPHERE, mujoco.mjtWrap.mjWRAP_CYLINDER):
                bodies.add(int(m.geom_bodyid[m.wrap_objid[k]]))
        s = set()
        for j in info.joint_ids:
            inside = sum(int(m.jnt_bodyid[j]) in anc[b] for b in bodies)
            if 0 < inside < len(bodies):
                s.add(m.joint(info.coupled[j][0] if j in info.coupled else j).name)
        spans[m.actuator(a).name] = s
    return spans


def get_mshuman700_cfg(self_collisions: bool = False, lock_hands: bool = False) -> EntityCfg:
    """Free-floating MS-Human-700. A muscle's ``ctrl`` is its excitation in [0, 1], written as the
    effort target of the muscle's tendon (see the task's muscle excitation action).

    Args:
      self_collisions: keep robot-robot contacts (off by default, as in SuperMimic).
      lock_hands: delete the hand + wrist DoF and the muscles acting only on them (see ``_lock_hands``).
    """
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=INIT_POS, rot=INIT_ROT, joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
        spec_fn=functools.partial(get_spec, self_collisions=self_collisions, lock_hands=lock_hands),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                XmlActuatorCfg(target_names_expr=MUSCLE_TENDON_EXPR, transmission_type=TransmissionType.TENDON, command_field="effort"),
            ),
            soft_joint_pos_limit_factor=1.0,
        ),
    )


@functools.lru_cache(maxsize=2)
def compiled_model(lock_hands: bool = False) -> mujoco.MjModel:
    """The entity compiled on its own (no scene, no name prefix) for CPU-side kinematics and parameters."""
    return get_spec(lock_hands=lock_hands).compile()


@functools.lru_cache(maxsize=1)
def original_model() -> mujoco.MjModel:
    """The upstream model as shipped (6 pelvis DoF), to convert clips stored in its 85-D qpos layout."""
    return mujoco.MjModel.from_xml_path(str(require_mshuman700_xml()))


def root_offset(m: mujoco.MjModel) -> np.ndarray:
    """ROOT_OFFSET with R_canonical = R_pelvis @ ROOT_OFFSET = identity in the upright rest pose."""
    R_rest = np.zeros(9)
    mujoco.mju_quat2Mat(R_rest, m.body(ROOT_BODY).quat)
    return R_rest.reshape(3, 3).T.copy()


class ModelInfo:
    """Joint and muscle bookkeeping derived from the compiled entity model.

    * independent joints: every non-root joint that is not the driven side of a joint ``<equality>``
      (37: legs 2x7, spine 9, arms 2x7), in model order -- the joint coordinates of the state, of the
      reference motions and of the world model;
    * coupled joints: the 42 joints set by ``q_c = qpos0_c + poly(q_i - qpos0_i)`` from one independent joint;
    * muscles: in actuator order (= activation order), with the tendon each one drives.
    """

    def __init__(self, m: mujoco.MjModel):
        self.m = m
        assert m.jnt_type[m.body_jntadr[m.body(ROOT_BODY).id]] == mujoco.mjtJoint.mjJNT_FREE
        coupled = {}
        for e in range(m.neq):
            if m.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT or not m.eq_active0[e]:
                continue
            j1, j2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
            assert j2 >= 0, f"equality {m.eq(e).name} fixes a joint to a constant"
            coupled[j1] = (j2, m.eq_data[e, :5].copy())
        for j1, (j2, _) in coupled.items():
            assert j2 not in coupled, f"{m.joint(j1).name} is coupled to another coupled joint"
        self.coupled = coupled  # joint id -> (independent joint id, polycoef[5])
        self.joint_ids = [j for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
        self.joint_names = [m.joint(j).name for j in self.joint_ids]
        self.indep_ids = [j for j in self.joint_ids if j not in coupled]
        self.indep_names = [m.joint(j).name for j in self.indep_ids]
        self.indep_cols = [self.joint_ids.index(j) for j in self.indep_ids]  # positions among the non-root joints
        self.qpos0 = m.qpos0.copy()
        self.q_lo, self.q_hi = m.jnt_range[self.indep_ids].T.copy()
        assert (m.actuator_trntype == mujoco.mjtTrn.mjTRN_TENDON).all()
        assert (m.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE).all()
        self.num_muscles = m.nu
        self.muscle_names = [m.actuator(a).name for a in range(m.nu)]
        self.muscle_tendon_names = [m.tendon(int(t)).name for t in m.actuator_trnid[:, 0]]

    def full_joint_pos(self, q: np.ndarray, qd: np.ndarray | None = None):
        """Independent coordinates (..., 37) -> every non-root joint (..., 79) in model order (numpy)."""
        m, col = self.m, {j: i for i, j in enumerate(self.indep_ids)}
        out_q = np.zeros(q.shape[:-1] + (len(self.joint_ids),))
        out_qd = np.zeros_like(out_q)
        for k, j in enumerate(self.joint_ids):
            if j in col:
                out_q[..., k] = q[..., col[j]]
                if qd is not None:
                    out_qd[..., k] = qd[..., col[j]]
                continue
            j2, c = self.coupled[j]
            d = q[..., col[j2]] - self.qpos0[m.jnt_qposadr[j2]]
            out_q[..., k] = self.qpos0[m.jnt_qposadr[j]] + c[0] + c[1] * d + c[2] * d**2 + c[3] * d**3 + c[4] * d**4
            if qd is not None:
                out_qd[..., k] = (c[1] + 2 * c[2] * d + 3 * c[3] * d**2 + 4 * c[4] * d**3) * qd[..., col[j2]]
        return out_q, out_qd
