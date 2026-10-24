"""Reference motion library: clips resampled to the control rate, with FK of the tracked bodies.

Port of supermimic_vibed's ``MotionLib`` (github.com/zita-ch/supermimic_vibed, MIT) for MS-Human-700.
A clip is one ``.npz`` in either format:

* **canonical** -- ``fps``, ``root_pos`` (T,3), ``root_quat`` (T,4, wxyz, canonical root frame: x
  forward, y left, z up), ``joint_pos`` (T,J) and ``joint_names`` (J,); without names the 37 columns
  are taken in the full model's order. Joints the model lacks (e.g. locked wrists) are dropped;
* **MS-Human-700 native** -- ``qpos_traj`` (T,85) in the upstream model's qpos layout (6 OpenSim
  pelvis DoF first) and ``framerate``, as msgym's ``motion_data/*.npz``. The root pose is recovered
  with MuJoCo FK on the upstream model; coupled joints are ignored (they follow from the independent ones).
"""

from __future__ import annotations

import glob
import os

import mujoco
import numpy as np
import torch

from muscle_supermimic.assets.robots.mshuman700 import ROOT_BODY, ModelInfo, compiled_model, original_model
from muscle_supermimic.utils.rotations import mat_to_quat, quat_slerp, quat_to_mat, so3_log

from .kinematics import MSHumanKinematics


def load_clip(path: str, info: ModelInfo, R_off: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """-> (fps, root_pos (T,3), root_quat (T,4) canonical wxyz, joint_pos (T,37) in ``info.indep_names`` order)."""
    data = np.load(path, allow_pickle=False)
    if "qpos_traj" in data.files:
        return (float(data["framerate"]),) + native_to_canonical(data["qpos_traj"], info, R_off)
    q = np.asarray(data["joint_pos"], dtype=np.float64)
    names = [str(n) for n in data["joint_names"]] if "joint_names" in data.files else ModelInfo(compiled_model()).indep_names
    assert q.shape[1] == len(names), f"{path}: joint_pos has {q.shape[1]} columns for {len(names)} joint names"
    missing = [n for n in info.indep_names if n not in names]
    assert not missing, f"{path}: no data for joints {missing}"
    q = q[:, [names.index(n) for n in info.indep_names]]
    return float(data["fps"]), np.asarray(data["root_pos"], np.float64), np.asarray(data["root_quat"], np.float64), q


def native_to_canonical(qpos: np.ndarray, info: ModelInfo, R_off: np.ndarray):
    """Upstream 85-D qpos frames -> canonical root pose + independent joint angles."""
    m = original_model()
    d = mujoco.MjData(m)
    pelvis = m.body(ROOT_BODY).id
    adr = [m.jnt_qposadr[m.joint(n).id] for n in info.indep_names]
    p, quat = np.zeros((len(qpos), 3)), np.zeros((len(qpos), 4))
    for t, x in enumerate(qpos):
        d.qpos[:] = x
        mujoco.mj_kinematics(m, d)
        p[t] = d.xpos[pelvis]
        mujoco.mju_mat2Quat(quat[t], (d.xmat[pelvis].reshape(3, 3) @ R_off).reshape(-1))
    return p, quat, qpos[:, adr].astype(np.float64)


class MotionLib:
    def __init__(self, files: list[str], kin: MSHumanKinematics, control_dt: float, device: str, bin_frames: int = 50):
        assert files, "no motion clips"
        self.device, self.dt, self.bin_frames = device, control_dt, bin_frames
        info, R_off = kin.info, kin.R_off.double().cpu().numpy()
        self.names = [os.path.splitext(os.path.basename(f))[0] for f in files]
        ps, qs, js, lens = [], [], [], []
        for f in files:
            fps, p, quat, j = load_clip(f, info, R_off)
            p, quat, j = (torch.tensor(x, dtype=torch.float32) for x in (p, quat, j))
            # resample to the control rate
            T_src = p.shape[0]
            dur = (T_src - 1) / fps
            n = int(np.floor(dur / control_dt + 1e-6)) + 1
            t = torch.arange(n, dtype=torch.float32) * control_dt * fps
            i0 = torch.clamp(t.floor().long(), max=T_src - 2)
            a = (t - i0.float()).clamp(0, 1)
            p = p[i0] * (1 - a[:, None]) + p[i0 + 1] * a[:, None]
            j = j[i0] * (1 - a[:, None]) + j[i0 + 1] * a[:, None]
            quat = quat_slerp(quat[i0], quat[i0 + 1], a)
            ps.append(p), qs.append(quat), js.append(j)
            lens.append(n)
        self.clip_len = torch.tensor(lens, device=device)
        self.clip_start = torch.cumsum(torch.tensor([0] + lens[:-1], dtype=torch.long), 0).to(device)
        self.clip_end = self.clip_start + self.clip_len - 1  # last valid frame (inclusive)
        self.num_frames = int(sum(lens))
        self.clip_id = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(lens)]).to(device)

        root_p, root_R, root_v, root_w, jq, jqd = [], [], [], [], [], []
        for p, quat, j in zip(ps, qs, js):
            R = quat_to_mat(quat)
            v = _gradient(p, control_dt)
            qd = _gradient(j, control_dt)
            # angular velocity (world) by central differences of rotations
            w = torch.zeros_like(p)
            w[1:-1] = so3_log(R[2:] @ R[:-2].transpose(-1, -2)) / (2 * control_dt)
            w[0] = so3_log(R[1] @ R[0].T) / control_dt
            w[-1] = so3_log(R[-1] @ R[-2].T) / control_dt
            root_p.append(p), root_R.append(R), root_v.append(v), root_w.append(w), jq.append(j), jqd.append(qd)
        cat = lambda xs: torch.cat(xs).to(device)  # noqa: E731
        self.root_p, self.root_R, self.root_v, self.root_w = cat(root_p), cat(root_R), cat(root_v), cat(root_w)
        self.q, self.qd = cat(jq), cat(jqd)
        self.root_quat = mat_to_quat(self.root_R)
        with torch.no_grad():
            bp, bR, bv, bw = [], [], [], []
            for s in range(0, self.num_frames, 65536):
                e = min(s + 65536, self.num_frames)
                out = kin.forward(self.root_p[s:e], self.root_R[s:e], self.root_v[s:e], self.root_w[s:e], self.q[s:e], self.qd[s:e])
                bp.append(out[0]), bR.append(out[1]), bv.append(out[2]), bw.append(out[3])
            self.body_p, self.body_R, self.body_v, self.body_w = torch.cat(bp), torch.cat(bR), torch.cat(bv), torch.cat(bw)
        # 1 s bins for return-based reset sampling
        bin_clip, bin_start = [], []
        for c in range(len(lens)):
            for b in range(int(np.ceil(lens[c] / bin_frames))):
                bin_clip.append(c)
                bin_start.append(int(self.clip_start[c]) + b * bin_frames)
        self.bin_clip = torch.tensor(bin_clip, device=device)
        self.bin_start = torch.tensor(bin_start, device=device)
        self.num_bins = len(bin_start)
        self.frame_bin = torch.zeros(self.num_frames, dtype=torch.long, device=device)
        for b in range(self.num_bins):
            s = bin_start[b]
            e = min(s + bin_frames, int(self.clip_end[bin_clip[b]]) + 1)
            self.frame_bin[s:e] = b

    @staticmethod
    def from_path(path: str, kin: MSHumanKinematics, control_dt: float, device: str, bin_frames: int = 50) -> MotionLib:
        files = sorted(glob.glob(os.path.join(path, "*.npz"))) if os.path.isdir(path) else [path]
        return MotionLib(files, kin, control_dt, device, bin_frames)

    def ref_state(self, f: torch.Tensor):
        return self.root_p[f], self.root_R[f], self.root_v[f], self.root_w[f], self.q[f], self.qd[f]


def _gradient(x: torch.Tensor, dt: float) -> torch.Tensor:
    g = torch.zeros_like(x)
    g[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    g[0] = (x[1] - x[0]) / dt
    g[-1] = (x[-1] - x[-2]) / dt
    return g
