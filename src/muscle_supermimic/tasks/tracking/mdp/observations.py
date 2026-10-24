"""Observation terms, MuscleMimic's: the decoder sees the full state (root height and orientation, every joint
position and velocity, muscle length / velocity / force / excitation / activation, foot touch) and the
state-dependent part of GoalTrajMimic (root-relative site positions, rotations and velocities); the encoder
sees the reference-only part (concise 5-frame lookahead and the motion phase)."""

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


@analytic(TrackingHeads.root_pos)
def root_pos(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.root_pos(*_ctx(env, command_name))


@analytic(TrackingHeads.joint_pos)
def joint_pos(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.joint_pos(*_ctx(env, command_name))


@analytic(TrackingHeads.root_vel)
def root_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.root_vel(*_ctx(env, command_name))


@analytic(TrackingHeads.joint_vel)
def joint_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.joint_vel(*_ctx(env, command_name))


@analytic(TrackingHeads.muscle_length)
def muscle_length(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.muscle_length(*_ctx(env, command_name))


@analytic(TrackingHeads.muscle_velocity)
def muscle_velocity(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.muscle_velocity(*_ctx(env, command_name))


@analytic(TrackingHeads.muscle_force)
def muscle_force(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.muscle_force(*_ctx(env, command_name))


@analytic(TrackingHeads.muscle_excitation)
def muscle_excitation(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.muscle_excitation(*_ctx(env, command_name))


@analytic(TrackingHeads.muscle_activation)
def muscle_activation(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.muscle_activation(*_ctx(env, command_name))


@analytic(TrackingHeads.touch)
def touch(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.touch(*_ctx(env, command_name))


@analytic(TrackingHeads.site_rpos)
def site_rpos(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.site_rpos(*_ctx(env, command_name))


@analytic(TrackingHeads.site_rangles)
def site_rangles(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.site_rangles(*_ctx(env, command_name))


@analytic(TrackingHeads.site_rvel)
def site_rvel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.site_rvel(*_ctx(env, command_name))


@analytic(TrackingHeads.goal)
def motion_goal(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return TrackingHeads.goal(*_ctx(env, command_name))
