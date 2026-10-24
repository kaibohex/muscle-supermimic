"""Termination terms. Failure: |e_z| > 0.25 m or tilt error > 0.8 on the root (sigmoid-softened in
imagination with ``soft_width``); time-outs: episode length and the clip's last frame (bootstrapped)."""

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


@analytic(H.bad_root_height)
def bad_root_height(env: ManagerBasedRlEnv, command_name: str, threshold: float, soft_width: float) -> torch.Tensor:
    del soft_width  # imagination only
    return H.bad_root_height(*_ctx(env, command_name), threshold=threshold)


@analytic(H.bad_root_tilt)
def bad_root_tilt(env: ManagerBasedRlEnv, command_name: str, threshold: float, soft_width: float) -> torch.Tensor:
    del soft_width  # imagination only
    return H.bad_root_tilt(*_ctx(env, command_name), threshold=threshold)


def motion_end(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """The step reached the clip's last frame (a time-out, not a failure)."""
    cmd: MotionCommand = env.command_manager.get_term(command_name)
    m = cmd.motion
    return cmd.f + 1 >= m.clip_end[m.clip_id[cmd.f]]
