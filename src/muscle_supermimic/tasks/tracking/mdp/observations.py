"""Observation terms. SuperMimic's proprioception (q - q0, qd, IMU angular velocity, projected gravity,
a_{t-1}, reference root orientation relative to the robot) and its goal (K future reference frames)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .heads import StepCtx, TrackingHeads, analytic

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

    from .commands import MotionCommand


def _ctx(env: ManagerBasedRlEnv, command_name: str) -> tuple[TrackingHeads, StepCtx]:
    cmd: MotionCommand = env.command_manager.get_term(command_name)
    return cmd.heads, cmd.obs_ctx()


def _last_action(h: TrackingHeads, ctx: StepCtx) -> torch.Tensor:
    return ctx.a


@analytic(TrackingHeads.joint_pos)
def joint_pos(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.joint_pos(*_ctx(env, command_name))


@analytic(TrackingHeads.joint_vel)
def joint_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.joint_vel(*_ctx(env, command_name))


@analytic(TrackingHeads.root_ang_vel_b)
def root_ang_vel_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.root_ang_vel_b(*_ctx(env, command_name))


@analytic(TrackingHeads.projected_gravity)
def projected_gravity(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.projected_gravity(*_ctx(env, command_name))


@analytic(_last_action)
def last_action(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return _last_action(*_ctx(env, command_name))


@analytic(TrackingHeads.motion_ref_ori_b)
def motion_ref_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.motion_ref_ori_b(*_ctx(env, command_name))


@analytic(TrackingHeads.goal)
def motion_goal(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.goal(*_ctx(env, command_name))
