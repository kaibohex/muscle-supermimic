"""Analytic MDP heads shared by the simulator and the world model (SuperMimic paper Appendix A).

Port of supermimic_vibed's ``mdp.py``. The state tuple is
``S = (p, R, v, w, q, qd, act)``: pelvis position, canonical root rotation matrix, world linear and
angular velocity of the pelvis link frame, the 37 independent joint positions and velocities, and the
700 muscle activations (the plant's actuator state). Every observation, reward and termination term
of the mjlab task is a function of a :class:`StepCtx` built on such a state, so the same code labels
simulator transitions (through the managers) and imagined ones (in the SuperMimic runner).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from muscle_supermimic.utils.rotations import mat_to_6d, mat_to_quat, quat_to_mat, rot_angle_sq, yaw_mat

from .kinematics import MSHumanKinematics
from .motion import MotionLib

BODY_STATE_DIM_EXTRA = 3 + 4 + 3 + 3  # p, quat, v, w (+ q, qd)


@dataclass
class StepCtx:
    """What an analytic term may look at.

    ``S`` is the state, ``f`` the reference frame it is compared with, ``a`` the action that led to
    ``S`` (the "last action" of an observation) and ``a_prev`` the action before it.
    """

    S: tuple
    f: torch.Tensor
    a: torch.Tensor | None = None
    a_prev: torch.Tensor | None = None
    cache: dict = field(default_factory=dict)


def flatten_body(S) -> torch.Tensor:
    p, R, v, w, q, qd = S[:6]
    return torch.cat((p, mat_to_quat(R), v, w, q, qd), -1)


def unflatten_body(x: torch.Tensor, nq: int):
    return x[..., 0:3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13], x[..., 13 : 13 + nq], x[..., 13 + nq : 13 + 2 * nq]


def invariant_features(S) -> torch.Tensor:
    """Heading- and position-invariant body-state features: height, tilt (6D), body-frame velocities, joints."""
    p, R, v, w, q, qd = S[:6]
    H = yaw_mat(R)
    Rt = R.transpose(-1, -2)
    return torch.cat(
        (
            p[..., 2:3],
            mat_to_6d(H.transpose(-1, -2) @ R),
            (Rt @ v.unsqueeze(-1)).squeeze(-1),
            (Rt @ w.unsqueeze(-1)).squeeze(-1),
            q,
            qd,
        ),
        -1,
    )


class TrackingHeads:
    def __init__(self, kin: MSHumanKinematics, motion: MotionLib, default_q: torch.Tensor, future_frames: int):
        self.kin, self.motion, self.default_q, self.K = kin, motion, default_q, future_frames
        self.nq = kin.num_q
        self.body_state_dim = BODY_STATE_DIM_EXTRA + 2 * self.nq
        self.inv_dim = 1 + 6 + 3 + 3 + 2 * self.nq
        self.num_tracked = len(kin.tracked_idx)

    # ---------------- shared pieces (cached per ctx) ----------------
    def fk(self, ctx: StepCtx):
        if "fk" not in ctx.cache:
            p, R, v, w, q, qd = ctx.S[:6]
            ctx.cache["fk"] = self.kin.forward(p, R, v, w, q, qd)
        return ctx.cache["fk"]

    def anchored_ref(self, ctx: StepCtx):
        """Reference bodies re-anchored to the robot's root (xy + yaw), as in BeyondMimic."""
        if "ref" not in ctx.cache:
            m, f = self.motion, ctx.f
            p, R = ctx.S[0], ctx.S[1]
            rp, rR = m.root_p[f], m.root_R[f]
            dR = yaw_mat(R @ rR.transpose(-1, -2))
            anchor = torch.cat((p[..., :2], rp[..., 2:3]), -1)
            ref_bp = anchor.unsqueeze(-2) + (dR.unsqueeze(-3) @ (m.body_p[f] - rp.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1)
            ref_bR = dR.unsqueeze(-3) @ m.body_R[f]
            ctx.cache["ref"] = (ref_bp, ref_bR)
        return ctx.cache["ref"]

    # ---------------- observation pieces (supermimic: q - q0, qd, IMU w, g, a_{t-1}, ref ori) ----------------
    def joint_pos(self, ctx: StepCtx) -> torch.Tensor:
        return ctx.S[4] - self.default_q

    def joint_vel(self, ctx: StepCtx) -> torch.Tensor:
        return ctx.S[5]

    def root_ang_vel_b(self, ctx: StepCtx) -> torch.Tensor:
        R, w = ctx.S[1], ctx.S[3]
        return (R.transpose(-1, -2) @ w.unsqueeze(-1)).squeeze(-1)

    def projected_gravity(self, ctx: StepCtx) -> torch.Tensor:
        return -ctx.S[1][..., 2, :]  # R^T (0, 0, -1)

    def motion_ref_ori_b(self, ctx: StepCtx) -> torch.Tensor:
        """Reference root orientation relative to the robot root (BeyondMimic's motion_anchor_ori_b)."""
        return mat_to_6d(ctx.S[1].transpose(-1, -2) @ self.motion.root_R[ctx.f])

    def goal(self, ctx: StepCtx) -> torch.Tensor:
        return self.motion.goal(ctx.f, self.K)

    def muscle_act(self, ctx: StepCtx) -> torch.Tensor:
        """Muscle activations, the plant state a rigid robot does not have (MuscleMimic observes it too)."""
        return ctx.S[6]

    # ---------------- critic ----------------
    def critic_features(self, ctx: StepCtx) -> torch.Tensor:
        """Critic input: invariant state (incl. muscle activations) and robot-vs-reference errors."""
        m, f = self.motion, ctx.f
        p, R, q, act = ctx.S[0], ctx.S[1], ctx.S[4], ctx.S[6]
        Ht = yaw_mat(R).transpose(-1, -2)
        bp = self.fk(ctx)[0]
        ref_bp = self.anchored_ref(ctx)[0]
        e_body = (Ht.unsqueeze(-3) @ (ref_bp - bp).unsqueeze(-1)).squeeze(-1).flatten(-2)
        e_root = (Ht @ (m.root_p[f] - p).unsqueeze(-1)).squeeze(-1)
        rel = mat_to_6d(R.transpose(-1, -2) @ m.root_R[f])
        return torch.cat((invariant_features(ctx.S), act, m.q[f] - q, e_body, e_root, rel), -1)

    def critic_dim(self, num_muscles: int) -> int:
        return self.inv_dim + num_muscles + self.nq + 3 * self.num_tracked + 3 + 6


# ---------------- analytic term implementations (signature: (heads, ctx, **params)) ----------------
def root_pos_exp(h: TrackingHeads, ctx: StepCtx, std: float) -> torch.Tensor:
    e = ((h.motion.root_p[ctx.f] - ctx.S[0]) ** 2).sum(-1)
    return torch.exp(-e / std**2)


def root_ori_exp(h: TrackingHeads, ctx: StepCtx, std: float) -> torch.Tensor:
    return torch.exp(-rot_angle_sq(h.motion.root_R[ctx.f], ctx.S[1]) / std**2)


def body_pos_exp(h: TrackingHeads, ctx: StepCtx, std: float) -> torch.Tensor:
    e = ((h.anchored_ref(ctx)[0] - h.fk(ctx)[0]) ** 2).sum(-1).mean(-1)
    return torch.exp(-e / std**2)


def body_ori_exp(h: TrackingHeads, ctx: StepCtx, std: float) -> torch.Tensor:
    e = rot_angle_sq(h.anchored_ref(ctx)[1], h.fk(ctx)[1]).mean(-1)
    return torch.exp(-e / std**2)


def body_lin_vel_exp(h: TrackingHeads, ctx: StepCtx, std: float) -> torch.Tensor:
    e = ((h.motion.body_v[ctx.f] - h.fk(ctx)[2]) ** 2).sum(-1).mean(-1)
    return torch.exp(-e / std**2)


def body_ang_vel_exp(h: TrackingHeads, ctx: StepCtx, std: float) -> torch.Tensor:
    e = ((h.motion.body_w[ctx.f] - h.fk(ctx)[3]) ** 2).sum(-1).mean(-1)
    return torch.exp(-e / std**2)


def action_rate_l2(h: TrackingHeads, ctx: StepCtx) -> torch.Tensor:
    return ((ctx.a - ctx.a_prev) ** 2).sum(-1)


def joint_pos_limits(h: TrackingHeads, ctx: StepCtx, soft_limit_factor: float) -> torch.Tensor:
    lo, hi = h.kin.q_lo, h.kin.q_hi
    mid, half = (lo + hi) / 2, (hi - lo) / 2 * soft_limit_factor
    q = ctx.S[4]
    return (torch.relu(q - (mid + half)) + torch.relu((mid - half) - q)).sum(-1)


def bad_root_height(h: TrackingHeads, ctx: StepCtx, threshold: float, soft_width: float | None = None) -> torch.Tensor:
    e_z = (h.motion.root_p[ctx.f][..., 2] - ctx.S[0][..., 2]).abs()
    return e_z > threshold if soft_width is None else torch.sigmoid((e_z - threshold) / soft_width)


def bad_root_tilt(h: TrackingHeads, ctx: StepCtx, threshold: float, soft_width: float | None = None) -> torch.Tensor:
    e_tilt = (h.motion.root_R[ctx.f][..., 2, 2] - ctx.S[1][..., 2, 2]).abs()
    return e_tilt > threshold if soft_width is None else torch.sigmoid((e_tilt - threshold) / soft_width)


def analytic(fn):
    """Attach the analytic implementation ``fn(heads, ctx, **params)`` to a manager term function, so the
    SuperMimic runner can evaluate the term on imagined states with the env cfg's own params."""

    def deco(term):
        term.analytic = fn
        return term

    return deco
