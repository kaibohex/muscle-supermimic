"""Differentiable forward kinematics of a musculoskeletal model in torch, built from the compiled MJCF.

Unlike the G1 (one hinge per link, at the link origin), the human skeletons have bodies with several
joints in sequence (MS-Human-700's knee carries 3 slides + 3 hinges), joints away from the body origin,
welded bodies (vertebrae, sternum) and joints that are polynomial functions of an independent joint
(``<equality>`` couplers, 42 in MS-Human-700). The FK here replicates ``mj_kinematics`` exactly for all
of that, takes only the independent coordinates, and propagates link-origin velocities alongside, so the
same function labels reference motions, simulator transitions and world-model predictions.

Only bodies on the path to a tracked body are evaluated; joint-less bodies on that path are folded
into their children's fixed offsets.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from muscle_supermimic.assets.robots.model_info import ModelInfo, root_offset
from muscle_supermimic.utils.rotations import axis_angle_to_mat, quat_to_mat


def _mv(R: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return (R @ x.unsqueeze(-1)).squeeze(-1)


class SkeletonKinematics:
    def __init__(self, tracked: list[str] | tuple[str, ...], device: str, m: mujoco.MjModel, dtype: torch.dtype = torch.float32):
        info = ModelInfo(m)
        self.info, self.device = info, device
        self.num_q = len(info.indep_ids)
        self.R_off = torch.tensor(root_offset(m), dtype=dtype, device=device)
        root = info.root_id
        tracked_ids = [m.body(n).id for n in tracked]
        # tracked bodies welded to the root (the root body itself, or a body without joints under it)
        self.tracked_on_root = [bool(m.body_weldid[b] == root) for b in tracked_ids]
        needed = set()
        for b in tracked_ids:
            while b != 0:
                needed.add(b)
                b = int(m.body_parentid[b])
        assert root in needed
        keep = {root} | set(tracked_ids) | {b for b in needed if m.body_jntnum[b] > 0}
        indep_col = {j: i for i, j in enumerate(info.indep_ids)}
        coupled_list = sorted(info.coupled)  # joint ids, evaluated once per forward
        coupled_col = {j: i for i, j in enumerate(coupled_list)}

        def body_offset(b):
            p, R = m.body_pos[b].copy(), np.zeros(9)
            mujoco.mju_quat2Mat(R, m.body_quat[b])
            return p, R.reshape(3, 3)

        # nodes in tree order (mujoco orders parents before children)
        self.body_names, self.parent, off_p, off_R, self.joints = [], [], [], [], []
        local = {}
        for b in range(1, m.nbody):
            if b not in keep:
                continue
            local[b] = len(self.body_names)
            self.body_names.append(m.body(b).name)
            if b == root:
                self.parent.append(-1)
                off_p.append(np.zeros(3)), off_R.append(np.eye(3))
                self.joints.append([])
                continue
            # fold the joint-less bodies between b and its nearest kept ancestor into b's offset
            chain, pb = [b], int(m.body_parentid[b])
            while pb not in local:
                assert m.body_jntnum[pb] == 0, f"{m.body(pb).name} has joints but was folded"
                chain.append(pb)
                pb = int(m.body_parentid[pb])
            p, R = np.zeros(3), np.eye(3)
            for c in reversed(chain):
                pc, Rc = body_offset(c)
                p, R = p + R @ pc, R @ Rc
            self.parent.append(local[pb])
            off_p.append(p), off_R.append(R)
            js = []
            for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
                jt = m.jnt_type[j]
                assert jt in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE), (m.joint(j).name, jt)
                src = ("q", indep_col[j]) if j in indep_col else ("c", coupled_col[j])
                jpos = m.jnt_pos[j].copy()
                js.append(
                    dict(
                        slide=jt == mujoco.mjtJoint.mjJNT_SLIDE,
                        axis=torch.tensor(m.jnt_axis[j], dtype=dtype, device=device),
                        pos=None if np.allclose(jpos, 0) else torch.tensor(jpos, dtype=dtype, device=device),
                        src=src,
                        qpos0=float(m.qpos0[m.jnt_qposadr[j]]),
                    )
                )
            self.joints.append(js)
        self.off_p = torch.tensor(np.array(off_p), dtype=dtype, device=device)
        self.off_R = torch.tensor(np.array(off_R), dtype=dtype, device=device)
        self.num_bodies = len(self.body_names)
        self.tracked_idx = [self.body_names.index(n) for n in tracked]
        # coupled joints: q_c = qpos0_c + sum_k c_k (q_i - qpos0_i)^k
        # (a joint held at a constant reads column 0 with zero higher coefficients, so it stays at qpos0 + c_0)
        srcs = [info.coupled[j][0] for j in coupled_list]
        self.c_src = torch.tensor([0 if s is None else indep_col[s] for s in srcs], device=device)
        self.c_coef = torch.tensor(np.array([info.coupled[j][1] for j in coupled_list]), dtype=dtype, device=device)
        self.c_q0 = torch.tensor([m.qpos0[m.jnt_qposadr[j]] for j in coupled_list], dtype=dtype, device=device)
        self.c_src_q0 = torch.tensor([0.0 if s is None else m.qpos0[m.jnt_qposadr[s]] for s in srcs], dtype=dtype, device=device)
        # every non-root joint in model order, as a gather from cat(independent, coupled)
        self.full_index = torch.tensor(
            [indep_col[j] if j in indep_col else self.num_q + coupled_col[j] for j in info.joint_ids], device=device
        )
        self.q_lo = torch.tensor(info.q_lo, dtype=dtype, device=device)
        self.q_hi = torch.tensor(info.q_hi, dtype=dtype, device=device)

    def coupled(self, q: torch.Tensor, qd: torch.Tensor):
        """Positions and velocities of the coupled joints, (..., 42) each."""
        d = q[..., self.c_src] - self.c_src_q0
        c = self.c_coef
        qc = self.c_q0 + c[:, 0] + d * (c[:, 1] + d * (c[:, 2] + d * (c[:, 3] + d * c[:, 4])))
        dqc = c[:, 1] + d * (2 * c[:, 2] + d * (3 * c[:, 3] + d * 4 * c[:, 4]))
        return qc, dqc * qd[..., self.c_src]

    def full_joint_state(self, q: torch.Tensor, qd: torch.Tensor):
        """Independent coordinates (..., 37) -> every non-root joint (..., 79) in model (= entity) order."""
        qc, qdc = self.coupled(q, qd)
        return torch.cat((q, qc), -1)[..., self.full_index], torch.cat((qd, qdc), -1)[..., self.full_index]

    def forward(self, p, R, v, w, q, qd, all_bodies: bool = False):
        """Tracked-body world positions, rotations, linear (link-origin) and angular velocities.

        Args:
            p, R, v, w: root position (B,3), canonical root rotation (B,3,3), world lin/ang velocity (B,3).
            q, qd: independent joint positions/velocities (B,37) in ``ModelInfo.indep_names`` order.
        """
        qc, qdc = self.coupled(q, qd)
        n = self.num_bodies
        P, Rs, V, W = [None] * n, [None] * n, [None] * n, [None] * n
        for i in range(n):
            par = self.parent[i]
            if par < 0:
                P[i], Rs[i], V[i], W[i] = p, R @ self.R_off.T, v, w
                continue
            Rp, pp = Rs[par], P[par]
            pi = pp + _mv(Rp, self.off_p[i])
            Ri = Rp @ self.off_R[i]
            vi = V[par] + torch.cross(W[par], pi - pp, dim=-1)
            wi = W[par]
            for jt in self.joints[i]:
                kind, col = jt["src"]
                qj = (q[..., col] if kind == "q" else qc[..., col]) - jt["qpos0"]
                qdj = (qd[..., col] if kind == "q" else qdc[..., col]).unsqueeze(-1)
                axis_w = _mv(Ri, jt["axis"])
                if jt["slide"]:
                    dp = axis_w * qj.unsqueeze(-1)
                    pi = pi + dp
                    vi = vi + torch.cross(wi, dp, dim=-1) + axis_w * qdj
                    continue
                if jt["pos"] is None:
                    wi = wi + axis_w * qdj
                    Ri = Ri @ axis_angle_to_mat(jt["axis"], qj)
                    continue
                anchor = pi + _mv(Ri, jt["pos"])
                va = vi + torch.cross(wi, anchor - pi, dim=-1)
                wi = wi + axis_w * qdj
                Ri = Ri @ axis_angle_to_mat(jt["axis"], qj)
                pi = anchor - _mv(Ri, jt["pos"])
                vi = va + torch.cross(wi, pi - anchor, dim=-1)
            P[i], Rs[i], V[i], W[i] = pi, Ri, vi, wi
        idx = range(n) if all_bodies else self.tracked_idx
        return (
            torch.stack([P[i] for i in idx], -2),
            torch.stack([Rs[i] for i in idx], -3),
            torch.stack([V[i] for i in idx], -2),
            torch.stack([W[i] for i in idx], -2),
        )

    def root_body_quat(self, R: torch.Tensor) -> torch.Tensor:
        """Canonical root rotation -> root body rotation (the free joint's qpos quaternion is its wxyz)."""
        return R @ self.R_off.T

    def canonical_root(self, quat_body: torch.Tensor) -> torch.Tensor:
        """Free-joint quaternion (wxyz) of the root body -> canonical root rotation matrix."""
        return quat_to_mat(quat_body) @ self.R_off
