"""Fixed-size GPU replay buffer of simulator transitions, laid out as (time, env).

Port of supermimic_vibed's ``buffer.py``. Muscle-sized fields (700 activations, 700 excitations) are
stored in fp16, and the action before a transition is read from the previous row instead of stored.
"""

from __future__ import annotations

import torch


class ReplayBuffer:
    def __init__(self, T: int, N: int, body_dim: int, num_muscles: int, num_touch: int, device: str):
        self.T, self.N, self.device = T, N, device
        f16 = torch.float16
        self.fields = {
            "s": ((body_dim,), torch.float32),  # body state before the step
            "act": ((num_muscles,), f16),  # muscle activations before the step
            "touch": ((num_touch,), torch.float32),  # foot touch before the step
            "a": ((num_muscles,), f16),  # excitation applied during the step
            "s1": ((body_dim,), torch.float32),  # body state after the step (activations follow analytically)
            "touch1": ((num_touch,), torch.float32),  # foot touch after the step
            "r": ((), torch.float32),
            "f": ((), torch.long),
            "ep_step": ((), torch.int32),
            "done": ((), torch.bool),
        }
        self.data = {k: torch.zeros((T, N) + shape, dtype=dt, device=device) for k, (shape, dt) in self.fields.items()}
        self.ptr, self.size = 0, 0

    def add(self, tr: dict):
        for k in self.fields:
            self.data[k][self.ptr] = tr[k]
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        """Map age-ordered time index (0 = oldest) to the storage row."""
        return (self.ptr - self.size + age) % self.T

    def _get(self, age: torch.Tensor, env: torch.Tensor, keys) -> dict:
        rows = self._row(age)
        out = {k: self.data[k][rows, env] for k in keys}
        for k in ("act", "a"):
            if k in out:
                out[k] = out[k].float()
        return out

    def _last_action(self, age: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
        """The action that led into the state at ``age`` (zero at an episode start); needs age >= 1."""
        first = self.data["ep_step"][self._row(age), env] == 0
        prev = self.data["a"][self._row(age - 1), env].float()
        return torch.where(first.unsqueeze(-1), 0.0, prev)

    def sample_windows(self, B: int, k: int):
        """k-step windows; returns dict of (B, k, ...) tensors and a validity mask (B, k)."""
        t0 = torch.randint(0, self.size - k + 1, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        ages = t0.unsqueeze(-1) + torch.arange(k, device=self.device)
        out = self._get(ages, env.unsqueeze(-1).expand_as(ages), ("s", "act", "a", "s1", "touch1", "done"))
        # transition j is valid if no episode ended at transitions 0..j-1
        d = out["done"].float()
        out["mask"] = torch.cumprod(torch.cat((torch.ones(B, 1, device=self.device), 1 - d[:, :-1]), -1), -1)
        return out

    def sample_starts(self, B: int, hist: int):
        """Start points for imagination plus their observation history (oldest first, clamped to the episode)."""
        age = torch.randint(max(hist, 1), self.size, (B,), device=self.device)  # row 0 has no previous action
        env = torch.randint(0, self.N, (B,), device=self.device)
        keys = ("s", "act", "touch", "f", "ep_step")
        start = self._get(age, env, keys)
        start["last_a"] = self._last_action(age, env)
        back = torch.arange(hist - 1, -1, -1, device=self.device)  # hist-1 ... 0
        back = torch.minimum(back.unsqueeze(0), start["ep_step"].long().unsqueeze(-1))  # stay inside the episode
        hage = age.unsqueeze(-1) - back
        henv = env.unsqueeze(-1).expand_as(hage)
        hist_d = self._get(hage, henv, keys)
        hist_d["last_a"] = self._last_action(hage, henv)
        return start, hist_d
