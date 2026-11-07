"""DEP against depRL's controller, transcribed; its restart at a reset; the runner's DEP-RL switching."""

from collections import deque

import pytest
import torch

from muscle_supermimic.rl.supermimic.dep import DEP, switch


class DepRLDEP:
    """depRL's ``DEP`` (deprl/dep_controller.py) on a batch of envs, transcribed in float64: explicit C, deque buffer."""

    def __init__(self, n_env, n, kappa, tau, time_dist, bias_rate, s4avg):
        self.kappa, self.tau, self.time_dist, self.bias_rate, self.s4avg = kappa, tau, time_dist, bias_rate, s4avg
        self.M = -torch.eye(n, dtype=torch.float64).expand(n_env, n, n)
        self.C_norm = torch.zeros(n_env, n, n, dtype=torch.float64)
        self.Cb = torch.zeros(n_env, n, dtype=torch.float64)
        self.buffer, self.t, self.obs_smoothed, self.reg = deque(maxlen=200), 0, None, 1e-32

    def step(self, obs):
        if self.s4avg > 1 and self.t > 0:
            self.obs_smoothed = self.obs_smoothed + (obs - self.obs_smoothed) / self.s4avg
        else:
            self.obs_smoothed = obs
        self.buffer.append([self.obs_smoothed.clone(), None])
        if len(self.buffer) > 2 + self.time_dist:
            self._learn_controller()
        q = torch.einsum("ijk, ik->ij", self.C_norm, self.obs_smoothed)
        q = torch.einsum("ij, i->ij", q, 1.0 / (torch.linalg.norm(q, axis=-1) + self.reg))
        y = torch.clamp(torch.tanh(q * self.kappa + self.Cb), -1.0, 1.0)
        self.buffer[-1][1] = y.clone()
        self.t += 1
        return y

    def _learn_controller(self):
        C = torch.zeros_like(self.C_norm)
        for s in range(2, min(self.t - self.time_dist, self.tau)):
            x, xx = self.buffer[-s][0], self.buffer[-s - 1][0]
            xx_t = x if self.time_dist == 0 else self.buffer[-s - self.time_dist][0]
            xxx_t = self.buffer[-s - 1 - self.time_dist][0]
            mu = torch.einsum("ijk, ik->ij", self.M, x - xx)
            C += torch.einsum("ij, ik->ijk", mu, xx_t - xxx_t)
        R = torch.einsum("ijk, imk->ijm", C, self.M)
        factor = self.kappa / (torch.linalg.norm(R, axis=-1) + self.reg)
        self.C_norm = torch.einsum("ijk,ik->ijk", C, factor)
        yy = self.buffer[-2][1]
        self.Cb = self.Cb - (torch.clip(yy * self.bias_rate, -0.05, 0.05) + self.Cb * 0.001)


def readings(T: int, N: int, n: int, seed: int) -> torch.Tensor:
    """Smooth sensor trajectories (T, N, n): a sinusoid per sensor plus a little noise."""
    g = torch.Generator().manual_seed(seed)
    w, phase = 0.05 + 0.3 * torch.rand(N, n, generator=g), 6.3 * torch.rand(N, n, generator=g)
    t = torch.arange(T).view(T, 1, 1)
    return (torch.sin(w * t + phase) + 0.01 * torch.randn(T, N, n, generator=g)).double()


@pytest.mark.parametrize("time_dist, s4avg", [(3, 2), (0, 1)])
def test_dep_matches_deprl(time_dist, s4avg):
    kw = dict(kappa=3.0, tau=12, time_dist=time_dist, bias_rate=0.01, s4avg=s4avg)
    ref, dep = DepRLDEP(3, 6, **kw), DEP(3, 6, **kw)
    for x in readings(80, 3, 6, 0):
        torch.testing.assert_close(dep.step(x.float()), ref.step(x).float(), rtol=0, atol=2e-4)


def test_dep_restarts_an_env_at_its_reset():
    kw = dict(kappa=3.0, tau=12, time_dist=3, bias_rate=0.01, s4avg=2)
    X = readings(60, 2, 6, 1).float()
    a, b = DEP(2, 6, **kw), DEP(2, 6, **kw)
    for x in X[:30]:
        a.step(x)
    a.reset(torch.tensor([0]))
    b.Cb.copy_(a.Cb)  # the bias carries over a reset
    for x in X[30:]:
        ya, yb = a.step(x), b.step(x)
        torch.testing.assert_close(ya[0], yb[0])  # env 0 as from scratch
    assert not torch.allclose(ya[1], yb[1])  # env 1 kept its history


def test_dep_rl_switching():
    """depRL's StochSwitchDep per env, on device tensors (the collector's)."""
    left = torch.zeros(4, dtype=torch.long)
    episodes = torch.tensor([1, 1, 2, 3])  # env 3 is in its third episode: the policy's alone
    off = torch.tensor(False)
    uses = []
    for _ in range(8):
        use, left = switch(left, episodes, off, proba=1.0, length=3, test_every=3)
        uses.append(use)
    use = torch.stack(uses)
    # an intervention starts the step after its draw and lasts `length` steps
    once = torch.tensor([0, 1, 1, 1, 0, 1, 1, 1], dtype=torch.bool)
    assert all(torch.equal(use[:, i], once) for i in range(3)) and not use[:, 3].any()
    # the initial phase: DEP drives every env, the interventions' state stays as it was
    use, kept = switch(left, episodes, torch.tensor(True), proba=1.0, length=3, test_every=3)
    assert use.all() and torch.equal(kept, left)
    use, _ = switch(torch.zeros(4, dtype=torch.long), episodes, off, proba=0.0, length=3, test_every=3)
    assert not use.any()
