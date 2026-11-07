"""DEP (differential extrinsic plasticity, Der & Martius 2015) as DEP-RL explores with it (Schumacher et al., ICLR 2023).

Port of depRL's ``dep_controller.py`` (https://github.com/martius-lab/depRL, MIT), batched over the envs on the
GPU: one controller per env, sensing that env's muscle lengths and driving its muscles. The controller matrix is
recomputed every step from the recent sensor velocities,

    C = -sum_{l=1}^{tau-2} d_l d_{l+time_dist}^T,    d_l = x_l - x_{l+1},

with x_l the smoothed sensor reading l steps ago (depRL's inverse model M stays -I), and normalized as depRL's
``independent`` normalization does it: column k scaled by kappa / |row k| (depRL's ``einsum("ijk,ik->ijk")``; DEP's
own scales rows). The action is y = tanh(kappa q / |q| + C_b), q = C_norm x, with a slowly adapting bias C_b.

C has rank <= tau - 2, so it is never formed: q and the row norms of C come from the velocities and their Gram
matrix (depRL forms the num_muscles^2 matrix of every env, 8 GB at 4096 envs x 700 muscles). Two changes for many
envs that reset on their own: an env's controller restarts at its reset (no velocity across the reset, no smoothing
over it; depRL's one buffer runs across episodes), and the sensors are normalized per muscle over all envs (depRL's
per env).

Every state tensor is updated in place and the buffer's write position lives on the device, so a step can be captured
in a CUDA graph (the collector's, collect_graph) and replayed; :func:`switch` is depRL's ``StochSwitchDep`` on device
tensors for the same reason.
"""

from __future__ import annotations

import torch

REG = 1e-32  # depRL's 10^-regularization, regularization = 32


def excitation(y: torch.Tensor) -> torch.Tensor:
    """DEP's output in [-1, 1] as muscle excitations: MyoSuite's ``normalize_act``, through which depRL's DEP drives
    MyoSuite's muscles (-1 -> 0.06 %, 0 -> 7.6 %, 1 -> 92.4 %)."""
    return torch.sigmoid(5.0 * (y - 0.5))


class DEP:
    def __init__(self, num_envs: int, num_muscles: int, kappa: float = 1169.7, tau: int = 40, time_dist: int = 5,
                 bias_rate: float = 0.002, s4avg: int = 2, force_scale: float = 0.0, device: str = "cpu"):  # fmt: skip
        assert tau >= 3 and time_dist >= 0
        self.kappa, self.td, self.bias_rate, self.s4avg, self.force_scale = kappa, time_dist, bias_rate, s4avg, force_scale
        self.S = tau - 2  # correlation terms
        N, n, dev = num_envs, num_muscles, device
        self.vel = torch.zeros(N, self.S + time_dist + 1, n, device=dev)  # smoothed-sensor velocities, newest at ptr
        self.ptr = torch.zeros((), dtype=torch.long, device=dev)
        self.lags = torch.arange(1, self.S + time_dist + 1, device=dev)
        self.x = torch.zeros(N, n, device=dev)  # smoothed sensors
        self.y = torch.zeros(N, n, device=dev)  # last output
        self.Cb = torch.zeros(N, n, device=dev)  # controller bias
        self.t = torch.zeros(N, dtype=torch.long, device=dev)  # steps since the controller (re)started (depRL's t)
        self.len_lo, self.force_lo = (torch.full((n,), torch.inf, device=dev) for _ in range(2))
        self.len_hi, self.force_hi = (torch.full((n,), -torch.inf, device=dev) for _ in range(2))

    def reset(self, env_ids: torch.Tensor):
        self.t[env_ids] = 0

    def reset_masked(self, done: torch.Tensor):
        """reset() of the envs where ``done`` (N,) is set, without a host synchronization."""
        self.t.masked_fill_(done, 0)

    def sensors(self, length: torch.Tensor, force: torch.Tensor) -> torch.Tensor:
        """depRL's DEP input (``muscle_states``) from muscle lengths and forces (N, n), each min-max normalized over
        the readings so far: 2 ((L - min) / (max - min + 0.1) - 0.5) + force_scale (F - min) / (max - min + 0.1)."""
        self.len_lo.copy_(torch.minimum(self.len_lo, length.amin(0)))
        self.len_hi.copy_(torch.maximum(self.len_hi, length.amax(0)))
        x = 2 * ((length - self.len_lo) / (self.len_hi - self.len_lo + 0.1) - 0.5)
        if self.force_scale > 0:
            self.force_lo.copy_(torch.minimum(self.force_lo, force.amin(0)))
            self.force_hi.copy_(torch.maximum(self.force_hi, force.amax(0)))
            x = x + self.force_scale * (force - self.force_lo) / (self.force_hi - self.force_lo + 0.1)
        return x

    def step(self, obs: torch.Tensor) -> torch.Tensor:
        """One DEP step on the sensors (N, n): smooth, relearn C, act. Returns y in [-1, 1] (N, n)."""
        fresh = (self.t == 0).unsqueeze(-1)
        x = obs if self.s4avg <= 1 else torch.where(fresh, obs, self.x + (obs - self.x) / self.s4avg)
        S, td, K = self.S, self.td, self.vel.shape[1]
        self.ptr.copy_((self.ptr + 1) % K)
        self.vel.index_copy_(1, self.ptr.view(1), (x - self.x).unsqueeze(1))  # a fresh reading's velocity is never paired
        self.x.copy_(x)
        # pairs (chi_l, v_l) = (d_l, d_{l+td}), l = 1 .. tau-2, as far as the readings since the start reach (depRL's
        # sum over its buffer, l <= t - td - 2)
        lags = self.lags
        D = self.vel[:, (self.ptr - lags) % K]  # (N, S + td, n), lag 1 first
        valid = lags[:S] <= (self.t - td - 2).unsqueeze(-1)
        chi, v = D[:, :S] * valid.unsqueeze(-1), D[:, td:]
        G = v @ v.mT
        row2 = (chi * (G @ chi)).sum(1)  # |row k of C|^2 = chi_k^T G chi_k, (N, n)
        # floored at the Gram form's rounding: about 1e-5 of (sum_l |chi_lk| |v_l|)^2, which bounds |row k|^2
        a = torch.einsum("nlk,nl->nk", chi.abs(), v.norm(dim=-1))
        row = torch.maximum(row2, 1e-5 * a * a).sqrt()
        q = -(chi.mT @ (v @ (x / (row + REG)).unsqueeze(-1))).squeeze(-1)  # C_norm x / kappa = C (x / row norms)
        q = q / (q.norm(dim=-1, keepdim=True) + REG)
        learning = (self.t >= td + 2).unsqueeze(-1)
        self.Cb.copy_(torch.where(learning, self.Cb - ((self.y * self.bias_rate).clamp(-0.05, 0.05) + 0.001 * self.Cb), self.Cb))
        self.y.copy_(torch.tanh(self.kappa * q + self.Cb))
        self.t.add_(1)
        return self.y.clone()


def switch(left: torch.Tensor, episodes: torch.Tensor, init: torch.Tensor, proba: float, length: int, test_every: int):
    """depRL's ``StochSwitchDep`` per env, on device tensors: which envs DEP drives this step, and the updated steps left
    of their interventions. ``init`` (0-dim bool): DEP drives every env (the initial phase, before the policy's first
    update). Otherwise an env in an intervention continues it (it starts the step after its draw and lasts ``length``
    steps), others start one with probability ``proba``; every ``test_every``-th episode of an env (depRL's greedy
    episodes; 0: none) the policy acts alone."""
    use = left > 0
    start = torch.rand(left.shape, device=left.device) < proba
    new_left = torch.where(use, left - 1, start.long() * length)
    if test_every > 0:
        greedy = episodes % test_every == 0
        use, new_left = use & ~greedy, torch.where(greedy, 0, new_left)
    return use | init, torch.where(init, left, new_left)
