"""Retarget MyoFullBody clips onto MS-Human-700, e.g. MuscleMimic's GMR-retargeted AMASS motions.

  uv run ms-retarget-myofullbody data/mshuman700_kit data/musclemimic/kit_kinesis_train.txt data/musclemimic/kit_kinesis_test.txt

The two skeletons share their OpenSim origins. The legs (hip, knee, ankle, subtalar, mtp) and the shoulder,
elbow and forearm (elv_angle, shoulder_elv, shoulder_rot, elbow flexion, pro_sup) have the same joints about
the same axes, so these angles carry over unchanged and turn their segments exactly alike; positions differ
by the knee's translation curves (< 2 cm) and by where the shoulders sit on the thorax (~4 cm). The pelvis
frame, and with it the canonical root pose, is the same in both. The rest is fitted to segment rotations,
all frames of a clip at once:

* spine: MyoFullBody bends its five lumbar levels with three coupled DoF and keeps thorax and head rigid on
  L1; MS-Human-700 has L5_S1 and T12_L1 (three DoF each) and a neck. L5_S1 and T12_L1 are fitted to the
  thorax's rotation relative to the pelvis (a small penalty on their size splits it between them), the neck
  stays at 0;
* wrists: MyoFullBody turns the whole hand about deviation/flexion at the lunate; MS-Human-700's
  deviation/flexion turn the proximal row and, through the midcarpal couplers, the hand (about twice as
  far). Flexion turns the hand about the same axis in both and is fitted to the hand's rotation relative to
  the radius. Deviation stays at 0: MS-Human-700's axis is oblique and moves the hand the other way round,
  and in MuscleMimic's GMR clips MyoFullBody's deviation mostly sits at its limit (an IK artifact).

Last, the root is lowered by one height per clip so that the feet touch the ground as in the source: the
foot bones coincide, but MS-Human-700's foot collision capsules sit ~0.8 cm higher than MyoFullBody's.

Output: canonical clips (see :mod:`..mdp.motion`) at the source rate, named by their path relative to the
folder the inputs share; each input list is also written out as a list of the retargeted clips.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os

import mujoco
import numpy as np
import torch

from muscle_supermimic.assets.robots import mshuman700, myofullbody
from muscle_supermimic.assets.robots.model_info import ModelInfo, root_offset
from muscle_supermimic.tasks.tracking.mdp.kinematics import SkeletonKinematics
from muscle_supermimic.tasks.tracking.mdp.motion import load_clip, motion_files
from muscle_supermimic.utils.rotations import mat_to_quat, quat_to_mat

RENAMED = {"elbow_flexion_r": "elbow_flex_r", "elbow_flexion_l": "elbow_flex_l"}  # MS-Human-700 -> MyoFullBody
FITTED = ("L5_S1_FE", "L5_S1_LB", "L5_S1_AR", "T12_L1_FE", "T12_L1_LB", "T12_L1_AR", "flexion_r", "flexion_l")
HELD = ("T1_head_neck_FE", "T1_head_neck_LB", "T1_head_neck_AR", "deviation_r", "deviation_l")  # at 0
# rotations of a segment relative to a reference body that the fit matches: (MS-Human-700, MyoFullBody) pairs
MATCHED = (
    (("thoracic12", "pelvis"), ("thorax", "pelvis")),
    (("hand_r", "radius_r"), ("lunate_r", "radius_r")),
    (("hand_l", "radius_l"), ("lunate_l", "radius_l")),
)
REG = 1e-3  # weight of the fitted angles' size against the rotation error (both in rad^2)
FEET = ("calcn_r", "toes_r", "calcn_l", "toes_l")


def lowest_foot_point(m: mujoco.MjModel, p: np.ndarray, quat: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Height of the lowest point of the foot collision geoms per frame, for a canonical root pose and the
    independent joints of model ``m``."""
    info, R_off = ModelInfo(m), torch.tensor(root_offset(m))
    q_full = info.full_joint_pos(q)[0]
    quat_body = mat_to_quat(quat_to_mat(torch.tensor(quat)) @ R_off.T).numpy()
    geoms = [g for g in range(m.ngeom) if m.geom_contype[g] and m.body(m.geom_bodyid[g]).name in FEET]
    d, out = mujoco.MjData(m), np.zeros(len(p))
    for t in range(len(p)):
        d.qpos[:3], d.qpos[3:7], d.qpos[7:] = p[t], quat_body[t], q_full[t]
        mujoco.mj_kinematics(m, d)
        out[t] = min(d.geom_xpos[g, 2] - geom_half_height(m, d, g) for g in geoms)
    return out


def geom_half_height(m: mujoco.MjModel, d: mujoco.MjData, g: int) -> float:
    """Extent of geom g below its center along world z."""
    zrow, size = d.geom_xmat[g].reshape(3, 3)[2], m.geom_size[g]
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:  # axis = local z
        return abs(zrow[2]) * size[1] + size[0]
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
        return size[0]
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return float(np.sqrt(((zrow * size) ** 2).sum()))
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
        return float(np.abs(zrow) @ size)
    raise NotImplementedError(f"foot geom {m.geom(g).name} of type {m.geom_type[g]}")


class _Segments:
    """Rotations of segments relative to reference bodies, as the rotation away from the rest pose expressed in
    world axes at rest: Rref0 A A0^T Rref0^T with A the segment's orientation in the reference frame. It does not
    depend on how either body's frame is set (the two models' thorax frames differ by 20 deg, their arm frames by
    7 deg, while their rest poses agree), so both models' rotations compare as anatomical ones: the wrist
    flexion axes agree to ~2 deg this way, against 7 deg in the radius frames."""

    def __init__(self, m, pairs, device: str):
        bodies = list(dict.fromkeys(b for pair in pairs for b in pair))
        self.kin = SkeletonKinematics(bodies, device, m, dtype=torch.float64)
        self.seg = [bodies.index(s) for s, _ in pairs]
        self.ref = [bodies.index(r) for _, r in pairs]
        R0 = self._fk(torch.zeros(1, self.kin.num_q, dtype=torch.float64, device=device))
        self.Rref0 = R0[:, self.ref]
        self.A0 = R0[:, self.ref].transpose(-1, -2) @ R0[:, self.seg]

    def _fk(self, q):
        n, dev = q.shape[0], q.device
        z = torch.zeros(n, 3, dtype=q.dtype, device=dev)
        eye = torch.eye(3, dtype=q.dtype, device=dev).expand(n, 3, 3)
        return self.kin.forward(z, eye, z, z, q, torch.zeros_like(q))[1]

    def __call__(self, q):
        R = self._fk(q)
        A = R[:, self.ref].transpose(-1, -2) @ R[:, self.seg]
        return self.Rref0 @ A @ self.A0.transpose(-1, -2) @ self.Rref0.transpose(-1, -2)  # (T, K, 3, 3)


class Retargeter:
    def __init__(self, device: str = "cpu"):
        self.ms, self.mf = ms, mf = mshuman700.compiled_model(), myofullbody.compiled_model()
        self.ms_info, self.mf_info = ModelInfo(ms), ModelInfo(mf)
        self.mf_R_off = root_offset(mf)
        names = self.ms_info.indep_names
        self.transferred = [n for n in names if n not in FITTED and n not in HELD]
        missing = [n for n in self.transferred if RENAMED.get(n, n) not in self.mf_info.indep_names]
        assert not missing, f"no MyoFullBody counterpart for {missing}"
        self.dst = torch.tensor([names.index(n) for n in self.transferred], device=device)
        self.src = torch.tensor([self.mf_info.indep_names.index(RENAMED.get(n, n)) for n in self.transferred], device=device)
        self.fit = torch.tensor([names.index(n) for n in FITTED], device=device)
        t = lambda x: torch.tensor(x, dtype=torch.float64, device=device)  # noqa: E731
        self.lo, self.hi = t(self.ms_info.q_lo), t(self.ms_info.q_hi)
        self.ms_seg = _Segments(ms, [p for p, _ in MATCHED], device)
        self.mf_seg = _Segments(mf, [p for _, p in MATCHED], device)
        self.device = device

    def __call__(self, q_mf: np.ndarray, iters: int = 200) -> tuple[np.ndarray, dict]:
        """MyoFullBody independent joints (T, 31) -> MS-Human-700 independent joints (T, 37), and fit statistics."""
        q_src = torch.tensor(q_mf, dtype=torch.float64, device=self.device)
        q = torch.zeros(len(q_src), len(self.ms_info.indep_names), dtype=torch.float64, device=self.device)
        v = q_src[:, self.src]
        q[:, self.dst] = torch.minimum(torch.maximum(v, self.lo[self.dst]), self.hi[self.dst])
        clamped = float((q[:, self.dst] != v).float().mean())
        target = self.mf_seg(q_src)
        lo, hi = self.lo[self.fit], self.hi[self.fit]
        # fitted angles = lo + (hi - lo) sigmoid(u), starting from 0 (inside every range here)
        u = torch.logit(((0 - lo) / (hi - lo)).clamp(1e-3, 1 - 1e-3)).expand(len(q), -1).clone().requires_grad_()

        def full(u):
            x = lo + (hi - lo) * torch.sigmoid(u)
            qq = q.clone()
            qq[:, self.fit] = x
            return qq, x

        def loss_fn():
            opt.zero_grad()
            qq, x = full(u)
            err = ((self.ms_seg(qq) - target) ** 2).sum((-1, -2)).sum(-1)  # ||R - R*||_F^2 = 4 (1 - cos angle) per segment
            loss = (err + REG * (x**2).sum(-1)).sum()
            loss.backward()
            return loss

        opt = torch.optim.LBFGS([u], lr=1.0, max_iter=iters, tolerance_grad=1e-12, tolerance_change=1e-14,
                                line_search_fn="strong_wolfe")  # fmt: skip
        opt.step(loss_fn)
        with torch.no_grad():
            qq, _ = full(u)
            dR = self.ms_seg(qq).transpose(-1, -2) @ target
            angle = torch.rad2deg(torch.arccos(((dR.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)))  # (T, K)
        stats = {"clamped": clamped, **{f"{s}_deg": angle[:, k].cpu().numpy() for k, ((s, _), _) in enumerate(MATCHED)}}
        return qq.cpu().numpy(), stats

    def clip(self, p: np.ndarray, quat: np.ndarray, q_mf: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        """A MyoFullBody clip (canonical root pose, independent joints) -> MS-Human-700's root position and
        independent joints (the root rotation carries over), and fit statistics."""
        q, stats = self(q_mf)
        dz = np.median(lowest_foot_point(self.ms, p, quat, q) - lowest_foot_point(self.mf, p, quat, q_mf))
        stats["ground_offset_m"] = float(dz)
        return p - np.array([0.0, 0.0, dz]), q, stats


_worker: Retargeter | None = None


def _init_worker(device: str, threads: int | None = 1) -> None:
    global _worker
    if threads:
        torch.set_num_threads(threads)  # one per process when several run side by side
    _worker = Retargeter(device)


def _retarget_file(job: tuple[str, str]) -> dict:
    """Retarget one clip file to ``out`` (a worker's job); returns its statistics."""
    f, out = job
    rt = _worker
    fps, p_mf, quat, q_mf = load_clip(f, rt.mf_info, rt.mf_R_off)
    p, q, stats = rt.clip(p_mf, quat, q_mf)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, fps=fps, root_pos=p, root_quat=quat, joint_pos=q, joint_names=np.array(rt.ms_info.indep_names))
    stats["frames"] = len(q)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output", help="directory for the retargeted clips (and lists)")
    ap.add_argument("inputs", nargs="+", help="MyoFullBody clips: .npz files, directories or .txt lists")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=1, help="clips retargeted in parallel processes")
    args = ap.parse_args()

    per_input = [motion_files(p) for p in args.inputs]
    files = list(dict.fromkeys(f for fs in per_input for f in fs))
    base = os.path.commonpath([os.path.dirname(os.path.abspath(f)) for f in files])
    name = {f: os.path.splitext(os.path.relpath(os.path.abspath(f), base))[0] for f in files}
    jobs = [(f, os.path.join(args.output, name[f] + ".npz")) for f in files]
    worst = {}
    if args.workers > 1:
        pool = mp.get_context("spawn").Pool(args.workers, initializer=_init_worker, initargs=(args.device,))
        results = pool.imap(_retarget_file, jobs)
    else:
        _init_worker(args.device, threads=None)
        results = map(_retarget_file, jobs)
    for i, (f, stats) in enumerate(zip(files, results, strict=True)):
        errs = {k: v for k, v in stats.items() if k.endswith("_deg")}
        for k, v in errs.items():
            worst[k] = max(worst.get(k, 0.0), float(v.max()))
        print(f"[{i + 1}/{len(files)}] {name[f]}: {stats['frames']} frames, clamped {100 * stats['clamped']:.2f} %, "
              f"ground {100 * stats['ground_offset_m']:+.2f} cm, "
              + ", ".join(f"{k[:-4]} {v.mean():.2f}/{v.max():.2f} deg" for k, v in errs.items()), flush=True)  # fmt: skip
    for path, fs in zip(args.inputs, per_input, strict=True):
        if path.endswith(".txt"):
            with open(os.path.join(args.output, os.path.basename(path)), "w") as fh:
                fh.write(f"# {len(fs)} clips retargeted from {path} onto MS-Human-700 (ms-retarget-myofullbody)\n")
                fh.writelines(name[f] + ".npz\n" for f in fs)
    print("worst fit residuals:", ", ".join(f"{k[:-4]} {v:.2f} deg" for k, v in worst.items()))


if __name__ == "__main__":
    main()
