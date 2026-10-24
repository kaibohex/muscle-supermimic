"""Reward terms: MuscleMimic's MimicReward, one term per component (exp(-w_exp * mean squared error)), each
comparing the post-step state with reference frame f + 1. Its penalties are all off in MuscleMimic's
full-body configuration, so there are none."""

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


@analytic(H.mimic_qpos)
def mimic_qpos(env: ManagerBasedRlEnv, command_name: str, w_exp: float) -> torch.Tensor:
    return H.mimic_qpos(*_ctx(env, command_name), w_exp=w_exp)


@analytic(H.mimic_qvel)
def mimic_qvel(env: ManagerBasedRlEnv, command_name: str, w_exp: float) -> torch.Tensor:
    return H.mimic_qvel(*_ctx(env, command_name), w_exp=w_exp)


@analytic(H.mimic_root_pos)
def mimic_root_pos(env: ManagerBasedRlEnv, command_name: str, w_exp: float) -> torch.Tensor:
    return H.mimic_root_pos(*_ctx(env, command_name), w_exp=w_exp)


@analytic(H.mimic_root_vel)
def mimic_root_vel(env: ManagerBasedRlEnv, command_name: str, w_exp: float) -> torch.Tensor:
    return H.mimic_root_vel(*_ctx(env, command_name), w_exp=w_exp)


@analytic(H.mimic_rpos)
def mimic_rpos(env: ManagerBasedRlEnv, command_name: str, w_exp: float) -> torch.Tensor:
    return H.mimic_rpos(*_ctx(env, command_name), w_exp=w_exp)


@analytic(H.mimic_rquat)
def mimic_rquat(env: ManagerBasedRlEnv, command_name: str, w_exp: float) -> torch.Tensor:
    return H.mimic_rquat(*_ctx(env, command_name), w_exp=w_exp)


@analytic(H.mimic_rvel_rot)
def mimic_rvel_rot(env: ManagerBasedRlEnv, command_name: str, w_exp: float) -> torch.Tensor:
    return H.mimic_rvel_rot(*_ctx(env, command_name), w_exp=w_exp)


@analytic(H.mimic_rvel_lin)
def mimic_rvel_lin(env: ManagerBasedRlEnv, command_name: str, w_exp: float) -> torch.Tensor:
    return H.mimic_rvel_lin(*_ctx(env, command_name), w_exp=w_exp)
