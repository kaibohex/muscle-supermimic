"""``EntityCfg`` for MuscleMimic's free-floating MyoFullBody.

The upstream file is loaded through ``MjSpec`` (as MuscleMimic loads it) and changed in these ways:

1. the standalone-scene decoration (floor, light, skybox, keyframe) goes, since mjlab's ``SceneCfg``
   owns the ground and the solver budget;
2. the root's default pose turns to face +x (see ``INIT_ROT``); the root's pose is the free joint's, so
   only the default changes;
3. the 106 frame-velocity sensors go (nothing reads them); the four foot touch sensors stay;
4. optionally (default, as MuscleMimic's ``disable_fingers``), the finger joints and the muscles that
   move the fingers are deleted, with their tendons;
5. optionally (``self_collisions=False``), the explicit self-contact pairs are deleted. The robot's
   geoms only touch the ground otherwise (contype 1, conaffinity 0 upstream).

Otherwise muscles, tendons, ``<equality>`` constraints, contact pairs, joint damping/armature are upstream.
MuscleMimic also widens the muscles' ctrlrange to [-1, 1]; MuJoCo clamps a muscle's control to [0, 1]
anyway, and the task's excitation action is already in [0, 1], so that change is not needed here.
"""

from __future__ import annotations

import functools

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from muscle_supermimic.assets.robots.myofullbody.myofullbody_constants import (
    FINGER_JOINTS,
    FINGER_MUSCLES,
    FREE_JOINT,
    INIT_POS,
    INIT_ROT,
    MUSCLE_TENDON_EXPR,
    MYOFULLBODY_XML,
    ROOT_BODY,
)

SCENE_TEXTURES: tuple[str, ...] = ("texplane", "sky")
SCENE_MATERIALS: tuple[str, ...] = ("MatPlane",)


def _strip_scene(spec: mujoco.MjSpec) -> None:
    """Remove what upstream's ``musclemimic_scene.xml`` adds, and the sizes and timestep the task's
    ``SimulationCfg`` sets; mjlab's scene provides its own."""
    spec.njmax = -1
    spec.nconmax = -1
    spec.nuser_jnt = -1
    spec.option.timestep = mujoco.MjSpec().option.timestep
    spec.nkey = 0
    for key in list(spec.keys):
        spec.delete(key)
    for geom in list(spec.worldbody.geoms):  # the floor plane
        spec.delete(geom)
    for light in list(spec.worldbody.lights):
        spec.delete(light)
    for material in list(spec.materials):
        if material.name in SCENE_MATERIALS:
            spec.delete(material)
    for texture in list(spec.textures):
        if texture.name in SCENE_TEXTURES:
            spec.delete(texture)


def get_spec(self_collisions: bool = True, disable_fingers: bool = True) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(MYOFULLBODY_XML))
    _strip_scene(spec)
    assert spec.joints[0].name == FREE_JOINT and spec.joints[0].type == mujoco.mjtJoint.mjJNT_FREE
    root = spec.body(ROOT_BODY)
    root.pos, root.quat = INIT_POS, INIT_ROT
    for sensor in list(spec.sensors):
        if sensor.type != mujoco.mjtSensor.mjSENS_TOUCH:
            spec.delete(sensor)
    if not self_collisions:
        for pair in list(spec.pairs):
            spec.delete(pair)
    if disable_fingers:
        _disable_fingers(spec)
    return spec


def _disable_fingers(spec: mujoco.MjSpec) -> None:
    """MuscleMimic's ``disable_fingers``: delete the finger joints (each finger welded to its hand at q = 0)
    and the finger muscles with their tendons. Upstream selects the tendons by substring; deleting each
    removed muscle's own tendon selects the same ones (tests check it)."""
    tendons = {a.target for a in spec.actuators if a.name in FINGER_MUSCLES}
    for actuator in list(spec.actuators):
        if actuator.name in FINGER_MUSCLES:
            spec.delete(actuator)
    for tendon in list(spec.tendons):
        if tendon.name in tendons:
            spec.delete(tendon)
    for joint in list(spec.joints):
        if joint.name in FINGER_JOINTS:
            spec.delete(joint)


def get_myofullbody_cfg(self_collisions: bool = True, disable_fingers: bool = True) -> EntityCfg:
    """Free-floating MyoFullBody. A muscle's ``ctrl`` is its excitation in [0, 1], written as the effort
    target of the muscle's tendon (see the task's muscle excitation action).

    Args:
      self_collisions: keep upstream's explicit self-contact pairs (MuscleMimic keeps them).
      disable_fingers: delete the finger DoF and finger muscles (MuscleMimic's default).
    """
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=INIT_POS, rot=INIT_ROT, joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
        spec_fn=functools.partial(get_spec, self_collisions=self_collisions, disable_fingers=disable_fingers),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                XmlActuatorCfg(
                    target_names_expr=MUSCLE_TENDON_EXPR, transmission_type=TransmissionType.TENDON, command_field="effort"
                ),
            ),
            soft_joint_pos_limit_factor=1.0,
        ),
    )


@functools.lru_cache(maxsize=2)
def compiled_model(disable_fingers: bool = True) -> mujoco.MjModel:
    """The entity compiled on its own (no scene, no name prefix) for CPU-side kinematics and parameters."""
    return get_spec(disable_fingers=disable_fingers).compile()
