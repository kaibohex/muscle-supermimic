"""Render videos of the human model, one per clip.

* a rollout recorded by ``ms-eval --record`` (MS-Human-700, with or without locked hands, or MyoFullBody, told
  apart by the size of its qpos): the simulated human (solid, muscles drawn) with the reference motion overlaid
  as a translucent ghost;
* an MS-Human-700 motion clip (canonical or native .npz): kinematic replay of the reference alone, as the motion
  library resamples it to 50 Hz, with a foot-ground clearance report (penetration / floating).

  uv run ms-render traj.npz out_dir/ [--clips walk] [--seconds 10]
  uv run ms-render data/motions/walk.npz out_dir/ [--lock-hands]
"""

from __future__ import annotations

import argparse
import os

import mujoco
import numpy as np

from muscle_supermimic.assets.robots import mshuman700, myofullbody
from muscle_supermimic.assets.robots.mshuman700.mshuman700_cfg import get_spec as mshuman700_spec
from muscle_supermimic.assets.robots.myofullbody.myofullbody_cfg import get_spec as myofullbody_spec
from muscle_supermimic.tasks.tracking.scripts.retarget_myofullbody import geom_half_height
from muscle_supermimic.utils.rotations import mat_to_quat

GHOST_RGBA = (0.35, 0.6, 1.0, 0.3)
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


def build_scene(spec_fn, ghost: bool = True) -> mujoco.MjModel:
    scene = mujoco.MjSpec()
    scene.visual.global_.offwidth, scene.visual.global_.offheight = 1920, 1080
    scene.visual.headlight.ambient = [0.4, 0.4, 0.4]
    scene.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    tex = scene.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                            rgb1=[0.25, 0.27, 0.3], rgb2=[0.32, 0.34, 0.38], width=512, height=512)  # fmt: skip
    mat = scene.add_material(name="grid", texrepeat=[8, 8], reflectance=0.1)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    scene.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="grid")
    scene.worldbody.add_light(pos=[0, 0, 5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.6, 0.6, 0.6])
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


def render(m: mujoco.MjModel, qpos: np.ndarray, width: int, height: int, feet: list[int]) -> tuple[list, np.ndarray]:
    """Frames of a qpos sequence (T, m.nq), the camera following robot/pelvis; also the lowest point of
    the robot's foot collision geoms ``feet`` per frame [m]."""
    data = mujoco.MjData(m)
    pelvis = m.body("robot/pelvis").id
    renderer = mujoco.Renderer(m, height, width)
    cam = mujoco.MjvCamera()
    cam.type, cam.distance, cam.azimuth, cam.elevation = mujoco.mjtCamera.mjCAMERA_FREE, 3.2, 110.0, -12.0
    opt = mujoco.MjvOption()
    frames, foot_z, lookat = [], [], None
    for x in qpos:
        data.qpos[:] = x
        mujoco.mj_forward(m, data)
        foot_z.append(min(data.geom_xpos[g, 2] - geom_half_height(m, data, g) for g in feet))
        target = data.xpos[pelvis].copy()
        lookat = target if lookat is None else 0.9 * lookat + 0.1 * target
        cam.lookat[:] = (lookat[0], lookat[1], 0.9)
        renderer.update_scene(data, cam, opt)
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
    m = build_scene(lambda: mshuman700_spec(lock_hands=lock_hands), ghost=False)
    for c, name in enumerate(mot.names):
        s, n = int(mot.clip_start[c]), int(mot.clip_len[c])
        n = min(n, int(seconds * 50)) if seconds else n
        frames, foot_z = render(m, qpos[s : s + n], width, height, foot_geoms(m, entity))
        out = os.path.join(out_dir, f"{name}_reference.mp4")  # clip names can hold subfolders
        os.makedirs(os.path.dirname(out), exist_ok=True)
        mediapy.write_video(out, frames, fps=50)
        lo, med, hi = np.percentile(foot_z, [1, 50, 99])
        print(f"{out}: {n} frames ({n / 50:.1f} s); lowest foot point per frame [m]: 1% {lo:+.3f}, median {med:+.3f}, "
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
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()
    import mediapy

    os.makedirs(args.out_dir, exist_ok=True)
    d = np.load(args.input)
    if "ref_qpos" not in d.files:
        return replay_motion(args.input, args.out_dir, args.lock_hands, args.seconds, args.width, args.height)
    qpos, f, alive, ref = d["qpos"], d["f"], d["alive"], d["ref_qpos"]
    fps = float(d["fps"]) if "fps" in d.files else 50.0  # control rate of the rollout
    nq = qpos.shape[-1]
    variant = next(k for k, (_, model) in VARIANTS.items() if model().nq == nq)
    spec_fn, entity = VARIANTS[variant][0], VARIANTS[variant][1]()
    m = build_scene(spec_fn)
    assert m.nq == 2 * nq
    print(f"{variant} rollout")
    for c, name in enumerate(d["names"]):
        if args.clips and name not in args.clips:
            continue
        steps = np.nonzero(alive[:, c])[0]
        if args.seconds:
            steps = steps[: int(args.seconds * fps)]
        frames, _ = render(m, np.concatenate((qpos[steps, c], ref[f[steps, c]]), -1), args.width, args.height, foot_geoms(m, entity))
        out = os.path.join(args.out_dir, f"{name}.mp4")  # clip names can hold subfolders
        os.makedirs(os.path.dirname(out), exist_ok=True)
        mediapy.write_video(out, frames, fps=fps)
        fell = len(steps) < (d["clip_len"][c] - 1) and not args.seconds
        print(f"{out}: {len(frames)} frames ({len(frames) / fps:.1f} s){' -- fell' if fell else ''}")


if __name__ == "__main__":
    main()
