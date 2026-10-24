"""World model, policy (motion encoder + shared decoder), and critic (paper Table A.4).

Port of supermimic_vibed's ``networks.py`` (github.com/zita-ch/supermimic_vibed, MIT). The world model
predicts the body state; the muscle activations, the plant's actuator state, are integrated
analytically (``mdp.muscle``), and the world model's action input is their substep-mean over the step.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from muscle_supermimic.tasks.tracking.mdp.heads import invariant_features
from muscle_supermimic.utils.rotations import so3_exp, so3_log, yaw_mat


def mlp(inp: int, hidden: list[int], out: int, act=nn.ELU, layer_norm: bool = False) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Running mean/std normalizer; updated from data without gradients, frozen after ``max_count``."""

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count, self.eps = max_count, eps

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).float()
        if self.count >= self.max_count:
            return
        n = x.shape[0]
        m, v = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + n
        delta = m - self.mean
        self.mean += delta * n / tot
        self.var = (self.var * self.count + v * n + delta**2 * self.count * n / tot) / tot
        self.count = tot

    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self):
        return torch.sqrt(self.var + self.eps**2)


# bf16 autocast for MLP matmuls only (state integration, FK, rewards stay fp32)
AMP_DTYPE: torch.dtype | None = None


def run_mlp(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if AMP_DTYPE is None:
        return net(x)
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        return net(x).float()


class WorldModel(nn.Module):
    """Deterministic body dynamics S_{t+1} = f(S_t, act_t, act_{t+1}) predicting frame-local state increments,
    with an auxiliary head for the foot touch of S_{t+1} (as supermusclemimic: the activations before and after
    the step are inputs, contacts are the only observation the analytic heads cannot compute)."""

    def __init__(self, nq: int, num_muscles: int, num_touch: int, hidden=(2048, 2048, 2048)):
        super().__init__()
        self.nq, self.num_touch = nq, num_touch
        self.in_dim = 1 + 6 + 3 + 3 + 2 * nq + 2 * num_muscles
        self.out_dim = 3 + 3 + 3 + 3 + 2 * nq
        self.net = mlp(self.in_dim, list(hidden), self.out_dim + num_touch, act=nn.SiLU, layer_norm=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(self.out_dim, eps=1e-4)
        self.touch_norm = RunningNorm(num_touch)

    def inputs(self, S, act, act1):
        return torch.cat((invariant_features(S), act, act1), -1)

    @staticmethod
    def targets(S, S1):
        """Frame-local increments from S to S1 (the regression targets)."""
        p, R, v, w, q, qd = S[:6]
        p1, R1, v1, w1, q1, qd1 = S1[:6]
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                (H.transpose(-1, -2) @ (p1 - p).unsqueeze(-1)).squeeze(-1),
                so3_log(Rt @ R1),
                (Rt @ (v1 - v).unsqueeze(-1)).squeeze(-1),
                (Rt @ (w1 - w).unsqueeze(-1)).squeeze(-1),
                q1 - q,
                qd1 - qd,
            ),
            -1,
        )

    def step(self, S, act, act1):
        """-> (body state of S_{t+1}, its normalized touch prediction)."""
        p, R, v, w, q, qd = S[:6]
        y = run_mlp(self.net, self.in_norm(self.inputs(S, act, act1)))
        d = y[..., : self.out_dim] * self.out_norm.std() + self.out_norm.mean
        H = yaw_mat(R)
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, self.nq, self.nq), -1)
        p1 = p + (H @ dp.unsqueeze(-1)).squeeze(-1)
        R1 = R @ so3_exp(dth)
        v1 = v + (R @ dv.unsqueeze(-1)).squeeze(-1)
        w1 = w + (R @ dw.unsqueeze(-1)).squeeze(-1)
        return (p1, R1, v1, w1, q + dq, qd + dqd), y[..., self.out_dim :]

    def touch(self, touch_n: torch.Tensor) -> torch.Tensor:
        """Normalized touch prediction -> touch (non-negative)."""
        return torch.relu(touch_n * self.touch_norm.std() + self.touch_norm.mean)

    @staticmethod
    def state_error(S_hat, S):
        """Frame-local body-state error components (same layout as ``targets``)."""
        return WorldModel.targets(S, S_hat)


class Policy(nn.Module):
    """e = sigmoid(dec(enc(g), o_t)); deterministic, latent z in R^32 bounded by tanh.

    The decoder outputs excitation logits; the env action is the excitation in [0, 1].
    """

    def __init__(self, obs_dim: int, goal_dim: int, num_actions: int, enc_hidden=(2048, 1024, 512, 256),
                 dec_hidden=(2048, 1024, 512, 256), z_dim: int = 32, init_excitation: float = 0.1):
        super().__init__()
        self.enc = mlp(goal_dim, list(enc_hidden), z_dim)
        self.dec = mlp(z_dim + obs_dim, list(dec_hidden), num_actions)
        with torch.no_grad():
            self.dec[-1].weight.mul_(0.01)
            self.dec[-1].bias.fill_(math.log(init_excitation / (1 - init_excitation)))
        self.obs_norm = RunningNorm(obs_dim)
        self.goal_norm = RunningNorm(goal_dim)

    def logits(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """obs: (B, obs_dim) the decoder observation; goal: (B, goal_dim) the encoder's reference-only goal."""
        z = torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))
        return run_mlp(self.dec, torch.cat((z, self.obs_norm(obs)), -1))

    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(obs, goal))


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden=(256, 128)):
        super().__init__()
        self.net = mlp(in_dim, list(hidden), 1)
        self.norm = RunningNorm(in_dim)

    def forward(self, x):
        return run_mlp(self.net, self.norm(x)).squeeze(-1)
