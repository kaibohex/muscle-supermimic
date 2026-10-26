"""Render videos of the human model, one per clip.

* a rollout recorded by ``ms-eval --record`` (MS-Human-700, with or without locked hands, or MyoFullBody, told
  apart by the size of its qpos): the simulated human with its muscles drawn, each colored by its activation when the
  recording holds them (light grey at rest to deep red; ``--no-activation`` keeps the model's own color), optionally
  with the reference motion overlaid as a translucent ghost (``--ghost``);
* an MS-Human-700 motion clip (canonical or native .npz): kinematic replay of the reference alone, as the motion
  library resamples it to 50 Hz, with a foot-ground clearance report (penetration / floating).

Studio look, 50 fps: a white floor fading into a light sky, a near-vertical key light casting the contact shadow,
and a camera that follows the root from a fixed direction, facing the body (25 deg off its front) at the start.

  uv run ms-render traj.npz out_dir/ [--clips walk] [--seconds 10] [--ghost]
  uv run ms-render data/mshuman700_kit/314/walking_medium08_poses.npz out_dir/ [--lock-hands]
"""

from __future__ import annotations

import argparse
import os

import mujoco
import numpy as np

from muscle_supermimic.assets.robots import mshuman700, myofullbody
from muscle_supermimic.assets.robots.model_info import ModelInfo, root_offset
from muscle_supermimic.assets.robots.mshuman700.mshuman700_cfg import get_spec as mshuman700_spec
from muscle_supermimic.assets.robots.myofullbody.myofullbody_cfg import get_spec as myofullbody_spec
from muscle_supermimic.tasks.tracking.scripts.retarget_myofullbody import geom_half_height
from muscle_supermimic.utils.rotations import mat_to_quat

GHOST_RGBA = (0.35, 0.6, 1.0, 0.3)
SKY_TOP, SKY_HORIZON, FLOOR_RGBA = [0.95, 0.955, 0.96], [1.0, 1.0, 1.0], [0.985, 0.985, 0.985, 1.0]
FPS = 50
# muscles by activation: one hue, light to dark, linear in the activation up to ACT_FULL
ACT_REST, ACT_RED, ACT_FULL = np.array([0.82, 0.79, 0.78]), np.array([0.72, 0.03, 0.05]), 1.0
# entity variants a rollout can come from: (spec, compiled entity model)
VARIANTS = {
    "MS-Human-700": (lambda: mshuman700_spec(), lambda: mshuman700.compiled_model()),
    "MS-Human-700 locked hands": (lambda: mshuman700_spec(lock_hands=True), lambda: mshuman700.compiled_model(lock_hands=True)),
    "MyoFullBody": (lambda: myofullbody_spec(), lambda: myofullbody.compiled_model()),
}
FOOT_BODIES = ("calcn_r", "toes_r", "calcn_l", "toes_l")


def foot_geoms(m: mujoco.MjModel, entity: mujoco.MjModel) -> list[int]:
    """The scene's robot copies of the entity's foot collision geoms."""
    names = [entity.geom(g).name for g in range(entity.ngeom)
             if entity.geom_contype[g] and entity.body(entity.geom_bodyid[g]).name in FOOT_BODIES]  # fmt: skip
    return [m.geom(f"robot/{n}").id for n in names]


def build_scene(spec_fn, ghost: bool = False) -> mujoco.MjModel:
    scene = mujoco.MjSpec()
    v = scene.visual
    v.global_.offwidth, v.global_.offheight = 1920, 1080
    v.quality.shadowsize = 8192
    v.headlight.ambient, v.headlight.diffuse, v.headlight.specular = [0.42] * 3, [0.35] * 3, [0.05] * 3
    v.rgba.fog = v.rgba.haze = SKY_HORIZON + [1.0]
    v.map.fogstart, v.map.fogend = 3.0, 9.0  # in units of the model's extent: the floor fades into the sky
    scene.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                      rgb1=SKY_TOP, rgb2=SKY_HORIZON, width=512, height=3072)  # fmt: skip
    # some emission keeps the lit floor white; the shadow still shows
    scene.add_material(name="floor", rgba=FLOOR_RGBA, emission=0.22, reflectance=0.0, specular=0.0, shininess=0.0)
    scene.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="floor")
    scene.worldbody.add_light(pos=[0, 0, 6], dir=[0.08, 0.12, -1.0], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                              diffuse=[0.55] * 3, specular=[0.15] * 3, castshadow=True)  # fmt: skip
    for prefix, is_ghost in (("robot/", False), ("ref/", True))[: 2 if ghost else 1]:
        spec = spec_fn()
        for geom in spec.geoms:
            geom.contype = geom.conaffinity = 0
            if is_ghost:
                geom.rgba = GHOST_RGBA
        if is_ghost:
            for tendon in spec.tendons:
                tendon.rgba = [0, 0, 0, 0]
        scene.attach(spec, prefix=prefix, frame=scene.worldbody.add_frame())
    return scene.compile()


def muscle_tendons(m: mujoco.MjModel, entity: mujoco.MjModel) -> np.ndarray:
    """The tendons the robot's muscles pull on, in its actuator order (the robot is attached first)."""
    n = entity.nu
    assert (m.actuator_trntype[:n] == mujoco.mjtTrn.mjTRN_TENDON).all()
    return m.actuator_trnid[:n, 0]


def activation_rgb(act: np.ndarray) -> np.ndarray:
    """(n,) activations -> (n, 3) colors from ACT_REST (0) to ACT_RED (ACT_FULL and above)."""
    w = np.clip(act / ACT_FULL, 0.0, 1.0)[:, None]
    return ACT_REST + w * (ACT_RED - ACT_REST)


def render(m: mujoco.MjModel, entity: mujoco.MjModel, qpos: np.ndarray, width: int, height: int,
           feet: list[int], act: np.ndarray | None = None) -> tuple[list, np.ndarray]:  # fmt: skip
    """Frames of a qpos sequence (T, m.nq) at 50 fps, the camera following the root of ``entity`` (the robot's
    model) from a fixed direction, 25 deg off the body's front at the first frame; also the lowest point of the
    robot's foot collision geoms ``feet`` per frame [m]. With ``act`` (T, entity.nu), each muscle is colored by its
    activation."""
    data = mujoco.MjData(m)
    tendons = muscle_tendons(m, entity) if act is not None else None
    root = m.body(f"robot/{ModelInfo(entity).root_name}").id
    renderer = mujoco.Renderer(m, height, width)
    data.qpos[:] = qpos[0]
    mujoco.mj_kinematics(m, data)
    front = (data.xmat[root].reshape(3, 3) @ root_offset(entity))[:, 0]  # the canonical root frame's x: forward
    cam = mujoco.MjvCamera()
    cam.type, cam.distance, cam.elevation = mujoco.mjtCamera.mjCAMERA_FREE, 3.8, -14.0
    cam.azimuth = np.degrees(np.arctan2(front[1], front[0])) + 180.0 + 25.0  # looking back at the body's front
    opt = mujoco.MjvOption()
    frames, foot_z, lookat = [], [], None
    for i, x in enumerate(qpos):
        data.qpos[:] = x
        if act is not None:
            m.tendon_rgba[tendons, :3] = activation_rgb(act[i])
        mujoco.mj_forward(m, data)
        foot_z.append(min(data.geom_xpos[g, 2] - geom_half_height(m, data, g) for g in feet))
        target = data.xpos[root].copy()
        lookat = target if lookat is None else 0.92 * lookat + 0.08 * target
        cam.lookat[:] = (lookat[0], lookat[1], 0.85)
        renderer.update_scene(data, cam, opt)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        frames.append(renderer.render())
    return frames, np.array(foot_z)


def replay_motion(path: str, out_dir: str, lock_hands: bool, seconds: float | None, width: int, height: int):
    import mediapy

    from muscle_supermimic.tasks.tracking.mdp.kinematics import SkeletonKinematics
    from muscle_supermimic.tasks.tracking.mdp.motion import MotionLib

    entity = mshuman700.compiled_model(lock_hands=lock_hands)
    kin = SkeletonKinematics(("pelvis",), "cpu", entity)
    mot = MotionLib.from_path(path, kin, 0.02, "cpu")
    q_full, _ = kin.full_joint_state(mot.q, mot.qd)
    qpos = torch_cat(mot.root_p, mat_to_quat(mot.root_R @ kin.R_off.T), q_full)
    m = build_scene(lambda: mshuman700_spec(lock_hands=lock_hands))
    for c, name in enumerate(mot.names):
        s, n = int(mot.clip_start[c]), int(mot.clip_len[c])
        n = min(n, int(seconds * FPS)) if seconds else n
        frames, foot_z = render(m, entity, qpos[s : s + n], width, height, foot_geoms(m, entity))
        out = os.path.join(out_dir, f"{name}_reference.mp4")  # clip names can hold subfolders
        os.makedirs(os.path.dirname(out), exist_ok=True)
        mediapy.write_video(out, frames, fps=FPS)
        lo, med, hi = np.percentile(foot_z, [1, 50, 99])
        print(f"{out}: {n} frames ({n / FPS:.1f} s); lowest foot point per frame [m]: 1% {lo:+.3f}, median {med:+.3f}, "
              f"99% {hi:+.3f}; below -1 cm in {np.mean(foot_z < -0.01) * 100:.0f}% of frames, above +3 cm in "
              f"{np.mean(foot_z > 0.03) * 100:.0f}%")  # fmt: skip


def torch_cat(*xs) -> np.ndarray:
    import torch

    return torch.cat(xs, -1).double().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="a trajectory from ms-eval --record, or a motion clip")
    ap.add_argument("out_dir")
    ap.add_argument("--lock-hands", action="store_true", help="motion replay on the locked-hands model")
    ap.add_argument("--clips", nargs="*", default=None, help="clip names (default: all)")
    ap.add_argument("--seconds", type=float, default=None, help="render at most this long per clip")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--ghost", action="store_true", help="overlay the reference motion as a translucent ghost")
    ap.add_argument("--no-activation", action="store_true", help="keep the muscles' own color, not their activation")
    args = ap.parse_args()
    import mediapy

    os.makedirs(args.out_dir, exist_ok=True)
    d = np.load(args.input)
    if "ref_qpos" not in d.files:
        return replay_motion(args.input, args.out_dir, args.lock_hands, args.seconds, args.width, args.height)
    qpos, f, alive, ref = d["qpos"], d["f"], d["alive"], d["ref_qpos"]
    fps = float(d["fps"]) if "fps" in d.files else 50.0  # control rate of the rollout
    every = max(1, round(fps / FPS))  # rendered every this many control steps
    nq = qpos.shape[-1]
    variant = next(k for k, (_, model) in VARIANTS.items() if model().nq == nq)
    spec_fn, entity = VARIANTS[variant][0], VARIANTS[variant][1]()
    m = build_scene(spec_fn, ghost=args.ghost)
    assert m.nq == (2 if args.ghost else 1) * nq
    print(f"{variant} rollout")
    for c, name in enumerate(d["names"]):
        if args.clips and name not in args.clips:
            continue
        steps = np.nonzero(alive[:, c])[0]
        fell = len(steps) < (d["clip_len"][c] - 1) and not args.seconds
        if args.seconds:
            steps = steps[: int(args.seconds * fps)]
        steps = steps[::every]
        q = np.concatenate((qpos[steps, c], ref[f[steps, c]]), -1) if args.ghost else qpos[steps, c]
        act = d["act"][steps, c] if "act" in d.files and not args.no_activation else None
        frames, _ = render(m, entity, q, args.width, args.height, foot_geoms(m, entity), act)
        out = os.path.join(args.out_dir, f"{name}.mp4")  # clip names can hold subfolders
        os.makedirs(os.path.dirname(out), exist_ok=True)
        mediapy.write_video(out, frames, fps=fps / every)
        print(f"{out}: {len(frames)} frames ({len(frames) * every / fps:.1f} s){' -- fell' if fell else ''}")


if __name__ == "__main__":
    main()
