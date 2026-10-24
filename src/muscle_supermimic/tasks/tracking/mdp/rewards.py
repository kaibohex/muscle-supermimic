"""Reward terms: SuperMimic's BeyondMimic tracking reward (without the self-collision term), action rate
and joint-limit penalties. Each compares the post-step state with reference frame f + 1."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import heads as H
from .heads import analytic

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

    from .commands import MotionCommand


def _ctx(env: ManagerBasedRlEnv, command_name: str):
    cmd: MotionCommand = env.command_manager.get_term(command_name)
    return cmd.heads, cmd.transition_ctx()


@analytic(H.root_pos_exp)
def motion_root_pos_exp(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
    return H.root_pos_exp(*_ctx(env, command_name), std=std)


@analytic(H.root_ori_exp)
def motion_root_ori_exp(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
    return H.root_ori_exp(*_ctx(env, command_name), std=std)


@analytic(H.body_pos_exp)
def motion_body_pos_exp(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
    return H.body_pos_exp(*_ctx(env, command_name), std=std)


@analytic(H.body_ori_exp)
def motion_body_ori_exp(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
    return H.body_ori_exp(*_ctx(env, command_name), std=std)


@analytic(H.body_lin_vel_exp)
def motion_body_lin_vel_exp(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
    return H.body_lin_vel_exp(*_ctx(env, command_name), std=std)


@analytic(H.body_ang_vel_exp)
def motion_body_ang_vel_exp(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
    return H.body_ang_vel_exp(*_ctx(env, command_name), std=std)


@analytic(H.action_rate_l2)
def action_rate_l2(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return H.action_rate_l2(*_ctx(env, command_name))


@analytic(H.joint_pos_limits)
def joint_pos_limits(env: ManagerBasedRlEnv, command_name: str, soft_limit_factor: float) -> torch.Tensor:
    return H.joint_pos_limits(*_ctx(env, command_name), soft_limit_factor=soft_limit_factor)
