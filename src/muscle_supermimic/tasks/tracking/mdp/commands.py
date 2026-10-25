"""Motion command: the reference library, the analytic MDP heads and SuperMimic's reset sampling.

The term keeps ``f``, the reference frame of each env's *current* state (supermimic_vibed's
``TrackingSim.f``): reward and termination terms, which run between physics and the command update,
compare the post-step state with frame ``f + 1``; observations, which run after the update, use ``f``.

Episodes start at reference states drawn by return-based bin sampling (paper Appendix D.1: 50%
uniform over 1 s bins, 40% softmax(-25 G_b / G_max), 10% a slow average of that softmax). The reset
writes the reference state exactly -- no pose, velocity or joint noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from mjlab.managers.command_manager import CommandTerm
from mjlab.tasks.tracking.mdp import MotionCommandCfg as MjlabMotionCommandCfg

from muscle_supermimic.assets.robots.model_info import ModelInfo
from muscle_supermimic.utils.rotations import mat_to_quat, quat_to_mat

from .heads import StepCtx, TrackingHeads
from .kinematics import SkeletonKinematics
from .motion import MotionLib
from .muscle import MuscleActivation
from .muscle_geometry import MuscleGeometry

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        if not cfg.motion_file:
            raise ValueError("motion_file is empty: pass --env.commands.motion.motion-file <clip.npz | dir> (play: --motion-file)")
        self.robot: Entity = env.scene[cfg.entity_name]
        m = self.robot.cfg.spec_fn().compile()  # the entity's own variant (e.g. locked hands), unprefixed
        self.info = ModelInfo(m)
        if m.body_weldid[m.body(cfg.anchor_body_name).id] != self.info.root_id:
            raise ValueError(f"the reward re-anchors on the root: anchor_body_name must be welded to {self.info.root_name!r}")
        assert tuple(self.robot.joint_names) == tuple(self.info.joint_names), "entity joints differ from the model's"
        dev = self.device
        self.kin = SkeletonKinematics(cfg.body_names, dev, m)
        bin_frames = max(1, round(cfg.bin_seconds / env.step_dt))
        self.motion = MotionLib.from_path(cfg.motion_file, self.kin, env.step_dt, dev, bin_frames=bin_frames)
        self.muscle = MuscleActivation(m, env.cfg.decimation, env.physics_dt, dev)
        self.geo = MuscleGeometry(m, dev)
        self.heads = TrackingHeads(self.kin, self.motion, self.geo, cfg.lookahead, cfg.stride)
        # the shared pieces of each step's terms in one compiled call: observations with muscle geometry, rewards and
        # terminations without (eager, they launch tens of thousands of small kernels per step)
        comp = torch.compile if cfg.compile and str(self.device).startswith("cuda") else (lambda fn: fn)
        self._shared_obs = comp(lambda S: self.heads.precompute(S, muscles=True))
        self._shared_transition = comp(lambda S: self.heads.precompute(S, muscles=False))
        self.touch_sensor = env.scene[cfg.touch_sensor_name]

        # simulator addresses (global, from the scene's model)
        idx = self.robot.indexing
        self._free_q = idx.free_joint_q_adr.long()
        self._free_v = idx.free_joint_v_adr.long()
        cols = torch.tensor(self.info.indep_cols, device=dev)
        self._q_adr = idx.joint_q_adr.long()[cols]
        self._v_adr = idx.joint_v_adr.long()[cols]
        self._act_adr = torch.tensor(env.sim.mj_model.actuator_actadr[idx.ctrl_ids.cpu().numpy()], device=dev).long()
        assert len(self._act_adr) == self.info.num_muscles

        # episode bookkeeping
        self.f = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        self.start_bin = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        self._just_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
        self.ep_disc_ret = torch.zeros(self.num_envs, device=dev)
        self.ep_ret = torch.zeros(self.num_envs, device=dev)
        self.max_step_reward = sum(r.weight for r in env.cfg.rewards.values() if r is not None and r.weight > 0) * env.step_dt
        self.G_max = self.max_step_reward / (1 - cfg.gamma)
        nb = self.motion.num_bins
        self.bin_G = torch.zeros(nb, device=dev)
        self.bin_w_avg = torch.full((nb,), 1.0 / nb, device=dev)
        self._ctx_step, self._ctx = -1, None
        self._obs_ctx_step, self._obs_ctx = -1, None

    # ------------------------------------------------------------------ state
    @property
    def command(self) -> torch.Tensor:
        return self.heads.goal(StepCtx((), self.f))

    def read_state(self):
        """(p, R, v, w, q, qd, act, touch) straight from qpos/qvel/act and the foot contact sensor (no derived
        quantities, so never stale). Touch is the normal force of the ground on each foot body, which on flat
        ground is what a MuJoCo touch sensor around the foot reads."""
        d = self._env.sim.data
        root_q, root_v = d.qpos[:, self._free_q], d.qvel[:, self._free_v]
        Rb = quat_to_mat(root_q[:, 3:7])
        p = root_q[:, :3] - self._env.scene.env_origins
        w = (Rb @ root_v[:, 3:6].unsqueeze(-1)).squeeze(-1)  # free joint: angular velocity is body-local
        touch = self.touch_sensor.data.force[..., 2].abs().reshape(self.num_envs, -1)
        return (p, Rb @ self.kin.R_off, root_v[:, :3], w, d.qpos[:, self._q_adr], d.qvel[:, self._v_adr], d.act[:, self._act_adr],
                touch)  # fmt: skip

    def obs_ctx(self) -> StepCtx:
        """The current state against frame f, cached for all observation terms of one step: they share its
        kinematics and muscle geometry. A reset or the command update (which move f or the state) drop it."""
        step = self._env.common_step_counter
        if self._obs_ctx is None or self._obs_ctx_step != step:
            am = self._env.action_manager
            self._obs_ctx, self._obs_ctx_step = StepCtx(self.read_state(), self.f.clone(), am.action, am.prev_action), step
            self._obs_ctx.cache.update(self._shared_obs(self._obs_ctx.S))
        return self._obs_ctx

    def transition_ctx(self) -> StepCtx:
        """The post-step state against frame f + 1, cached for all reward/termination terms of one step."""
        step = self._env.common_step_counter
        if self._ctx_step != step:
            m = self.motion
            f1 = torch.minimum(self.f + 1, m.clip_end[m.clip_id[self.f]])
            am = self._env.action_manager
            self._ctx, self._ctx_step = StepCtx(self.read_state(), f1, am.action, am.prev_action), step
            self._ctx.cache.update(self._shared_transition(self._ctx.S))
        return self._ctx

    # ------------------------------------------------------------------ reset
    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        if isinstance(env_ids, torch.Tensor) and len(env_ids) > 0:
            self._finish_episodes(env_ids)
        return super().reset(env_ids)

    def _finish_episodes(self, ids: torch.Tensor) -> None:
        """Fold finished episodes into their start bin's return estimate; truncated ones bootstrap with
        their mean reward."""
        env, g = self._env, self.cfg.gamma
        T = env.episode_length_buf[ids].float()
        term = getattr(env, "reset_terminated", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))[ids]
        valid = T > 0
        boot = torch.where(term, 0.0, (g**T) * (self.ep_ret[ids] / T.clamp(min=1)) / (1 - g))
        G = self.ep_disc_ret[ids] + boot
        b = self.start_bin[ids][valid]
        self.bin_G.index_put_((b,), 0.8 * self.bin_G[b] + 0.2 * G[valid])
        self.ep_disc_ret[ids] = 0.0
        self.ep_ret[ids] = 0.0

    def sample_frames(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.motion
        soft = torch.softmax(-25.0 * self.bin_G / self.G_max, 0)
        self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
        if self.cfg.sampling_mode == "uniform":
            probs = torch.ones(m.num_bins, device=self.device)
        else:
            probs = 0.5 / m.num_bins + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        bins = torch.multinomial(probs, n, replacement=True)
        start = m.bin_start[bins]
        end = torch.minimum(start + m.bin_frames, m.clip_end[m.bin_clip[bins]])  # keep >= 1 step before the clip end
        f = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        return f, bins

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if self.cfg.sampling_mode == "start":
            frames = self.motion.clip_start[env_ids % len(self.motion.names)]
            bins = self.motion.frame_bin[frames]
        else:
            frames, bins = self.sample_frames(len(env_ids))
        self.write_reference_state(env_ids, frames)
        self.start_bin[env_ids] = bins

    def write_reference_state(self, env_ids: torch.Tensor, frames: torch.Tensor) -> None:
        p, R, v, w, q, qd = (x.clone() for x in self.motion.ref_state(frames))
        q = torch.clamp(q, self.kin.q_lo, self.kin.q_hi)
        q_full, qd_full = self.kin.full_joint_state(q, qd)
        quat = mat_to_quat(R @ self.kin.R_off.T)
        root = torch.cat((p + self._env.scene.env_origins[env_ids], quat, v, w), -1)
        self.robot.write_root_state_to_sim(root, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(q_full, qd_full, env_ids=env_ids)
        self._env.sim.data.act[env_ids[:, None], self._act_adr] = self.cfg.init_activation
        self.f[env_ids] = frames
        self._just_reset[env_ids] = True
        self._obs_ctx = None

    # ------------------------------------------------------------------ step
    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is not None:  # reset path: the reset frame stays
            self._just_reset[env_ids] = False
            return
        stepped = ~self._just_reset
        env = self._env
        r = torch.where(stepped, env.reward_buf, 0.0)
        T = env.episode_length_buf.float()
        self.ep_disc_ret += (self.cfg.gamma ** (T - 1).clamp(min=0)) * r
        self.ep_ret += r
        self.f += stepped.long()
        self._just_reset[:] = False
        self._obs_ctx = None

    def _update_metrics(self) -> None:
        pass


@dataclass(kw_only=True)
class MotionCommandCfg(MjlabMotionCommandCfg):
    """Subclasses mjlab's tracking command cfg so that mjlab's ``train`` (``--env.commands.motion.motion-file``)
    and ``play`` (``--motion-file``) CLIs hand it the motion. Of the inherited fields only ``motion_file``,
    ``anchor_body_name``, ``body_names`` and ``entity_name`` are used; the inherited reset randomization
    ranges must stay empty (this task does not randomize anything)."""

    motion_file: str = ""
    """A clip (.npz) or a directory of clips; see :mod:`.motion` for the formats."""
    anchor_body_name: str = "pelvis"
    """The root's rigid body (the root body itself, or one welded to it)."""
    body_names: tuple[str, ...] = ()
    """Tracked bodies of the reward (the root's rigid body first)."""
    entity_name: str = "robot"
    joint_position_range: tuple[float, float] = (0.0, 0.0)
    sampling_mode: Literal["return", "uniform", "start"] = "return"
    """Reset frames: SuperMimic's return-based bins, uniform bins, or each env at the start of clip env_id % C."""
    lookahead: int = 5
    """Reference frames in the goal (MuscleMimic's n_step_lookahead), the current one included."""
    stride: int = 20
    """Control steps between them (MuscleMimic's n_step_stride: 0.2 s at 100 Hz)."""
    touch_sensor_name: str = "foot_touch"
    """Scene contact sensor of the four foot bodies against the terrain (net force per body)."""
    gamma: float = 0.99
    bin_seconds: float = 1.0
    init_activation: float = 0.0
    """Muscle activation written at reset (MuJoCo's reset value is 0)."""
    compile: bool = True
    """torch.compile the kinematics and muscle geometry the terms share each step (CUDA only)."""
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    def __post_init__(self):
        if self.pose_range or self.velocity_range or any(self.joint_position_range):
            raise ValueError("reset randomization (pose_range / velocity_range / joint_position_range) is not supported")

    def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
        return MotionCommand(self, env)
