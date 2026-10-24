"""``EntityCfg`` for the free-floating MS-Human-700, plus the joint/muscle bookkeeping the task needs.

The upstream file is loaded through ``MjSpec`` and changed in these ways:

1. the six pelvis DoF become one free joint, so the root has no gimbal lock and its state is the
   (p, R, v, w) tuple SuperMimic works with (mjlab also requires a free joint for a floating base);
2. the standalone-scene decoration (floor, lights, cameras, keyframe, ``<size>`` hints) goes, since
   mjlab's ``SceneCfg`` owns the ground and the solver budget;
3. optionally (default), robot-robot contacts are disabled while robot-ground contacts stay, as in
   SuperMimic.

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


def get_spec(self_collisions: bool = False) -> mujoco.MjSpec:
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
    return spec


def get_mshuman700_cfg(self_collisions: bool = False) -> EntityCfg:
    """Free-floating MS-Human-700. A muscle's ``ctrl`` is its excitation in [0, 1], written as the
    effort target of the muscle's tendon (see the task's muscle excitation action).

    Args:
      self_collisions: keep robot-robot contacts (off by default, as in SuperMimic).
    """
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=INIT_POS, rot=INIT_ROT, joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
        spec_fn=functools.partial(get_spec, self_collisions=self_collisions),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                XmlActuatorCfg(target_names_expr=MUSCLE_TENDON_EXPR, transmission_type=TransmissionType.TENDON, command_field="effort"),
            ),
            soft_joint_pos_limit_factor=1.0,
        ),
    )


@functools.lru_cache(maxsize=1)
def compiled_model() -> mujoco.MjModel:
    """The entity compiled on its own (no scene, no name prefix) for CPU-side kinematics and parameters."""
    return get_spec().compile()


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
