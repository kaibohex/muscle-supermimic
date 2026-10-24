"""SuperMimic runner (paper Algorithm A.1): collect -> fit world model -> BPTT policy + critic update.

Port of supermimic_vibed's ``train.py`` onto an mjlab ``ManagerBasedRlEnv``, with mjlab's runner
interface (``learn`` / ``save`` / ``load`` / ``get_inference_policy``) so mjlab's ``train`` and ``play``
CLIs drive it. The env's reward, termination and observation terms are reproduced on imagined
states through their analytic implementations (``term.func.analytic``), with the term weights and
params of the env cfg, so simulator and world model share one MDP definition.

Imagined rollouts evolve the muscle activations with the exact activation dynamics and the body
state with the learned world model, whose action input is the substep-mean activation.
"""

from __future__ import annotations

import copy
import csv
import math
import os
import time
from dataclasses import asdict, fields

import torch
import torch.nn as nn
from tensordict import TensorDict

from muscle_supermimic.rl.supermimic import networks
from muscle_supermimic.rl.supermimic.buffer import ReplayBuffer
from muscle_supermimic.rl.supermimic.config import SuperMimicRunnerCfg
from muscle_supermimic.rl.supermimic.networks import Critic, Policy, WorldModel
from muscle_supermimic.tasks.tracking.mdp.heads import StepCtx, flatten_body, unflatten_body


class AnalyticObsGroup:
    """An observation group of the env, evaluated from states: same terms, order, history and scale."""

    def __init__(self, env, group: str, heads):
        om = env.observation_manager
        self.heads, self.terms = heads, []
        for name, cfg in zip(om._group_obs_term_names[group], om._group_obs_term_cfgs[group], strict=True):
            fn = getattr(cfg.func, "analytic", None)
            if fn is None:
                raise ValueError(f"observation term {group}/{name} has no analytic implementation")
            if (cfg.noise is not None and om.cfg[group].enable_corruption) or cfg.clip or cfg.delay_max_lag > 0:
                raise NotImplementedError(f"observation term {group}/{name}: noise/clip/delay are not reproduced in imagination")
            params = {k: v for k, v in cfg.params.items() if k != "command_name"}
            self.terms.append((fn, params, cfg.history_length, cfg.scale))
        self.dim = om.group_obs_dim[group][0]

    def _value(self, fn, params, scale, ctx):
        x = fn(self.heads, ctx, **params)
        return x * scale if scale is not None else x

    def init_history(self, ctx_hist: StepCtx) -> list:
        """Histories from states with leading dims (B, H), oldest first."""
        return [self._value(fn, p, s, ctx_hist)[:, -h:] if h > 0 else None for fn, p, h, s in self.terms]

    def push(self, hist: list, ctx: StepCtx) -> list:
        return [None if x is None else torch.cat((x[:, 1:], self._value(fn, p, s, ctx).unsqueeze(1)), 1)
                for x, (fn, p, h, s) in zip(hist, self.terms)]  # fmt: skip

    def assemble(self, hist: list | None, ctx: StepCtx) -> torch.Tensor:
        out = []
        for i, (fn, p, h, s) in enumerate(self.terms):
            out.append(hist[i].flatten(1) if h > 0 else self._value(fn, p, s, ctx))
        return torch.cat(out, -1)


class SuperMimicRunner:
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu", **kwargs):
        names = {f.name for f in fields(SuperMimicRunnerCfg)}
        self.cfg = cfg = SuperMimicRunnerCfg(**{k: v for k, v in train_cfg.items() if k in names})
        self.env, self.device, self.log_dir = env, device, log_dir
        self.uenv = u = env.unwrapped
        self.cmd = u.command_manager.get_term(cfg.command_name)
        self.heads, self.muscle, self.motion = self.cmd.heads, self.cmd.muscle, self.cmd.motion
        self.nq, self.nm = self.heads.nq, self.cmd.info.num_muscles
        self.obs_policy = AnalyticObsGroup(u, cfg.obs_groups["policy"][0], self.heads)
        self.obs_goal = AnalyticObsGroup(u, cfg.obs_groups["goal"][0], self.heads)
        self.hist = max(h for _, _, h, _ in self.obs_policy.terms)
        assert all(h in (0, self.hist) for _, _, h, _ in self.obs_policy.terms), "mixed history lengths"
        rm, scale = u.reward_manager, (u.step_dt if u.reward_manager._scale_by_dt else 1.0)
        self.reward_terms = []
        for name, rc in zip(rm._term_names, rm._term_cfgs, strict=True):
            fn = getattr(rc.func, "analytic", None)
            if fn is None:
                raise ValueError(f"reward term {name} has no analytic implementation")
            self.reward_terms.append((fn, rc.weight * scale, {k: v for k, v in rc.params.items() if k != "command_name"}))
        tm = u.termination_manager
        self.fail_terms = []
        for name, tc in zip(tm._term_names, tm._term_cfgs, strict=True):
            if tc.time_out:
                continue
            fn = getattr(tc.func, "analytic", None)
            if fn is None:
                raise ValueError(f"termination term {name} has no analytic implementation")
            self.fail_terms.append((fn, {k: v for k, v in tc.params.items() if k != "command_name"}))
        self.max_step_reward = self.cmd.max_step_reward
        self.reward_scale, self.reward_names = scale, list(rm._term_names)

        networks.AMP_DTYPE = torch.bfloat16 if cfg.amp == "bf16" and str(device).startswith("cuda") else None
        self._build_networks()
        self.it, self.wall_offset, self.sim_steps = 0, 0.0, 0
        # per-component floors on the error normalization (pos, rot, lin vel, ang vel, q, qd increments)
        f = lambda n, v: torch.full((n,), v)  # noqa: E731
        self.err_floor = torch.cat((f(3, 3e-3), f(3, 5e-3), f(3, 2e-2), f(3, 5e-2), f(self.nq, 3e-3), f(self.nq, 5e-2))).to(device)
        comp = cfg.compile and str(device).startswith("cuda")
        c = torch.compile if comp else (lambda fn: fn)
        self.wm_rollout, self.wm_err = c(self._wm_rollout), c(self._wm_err)
        self.imag_step, self.reward_term, self.critic_input = c(self._imag_step), c(self._reward_term), c(self._critic_input)

    NET_FIELDS = ("wm_hidden", "enc_hidden", "dec_hidden", "latent_dim", "critic_hidden")

    def _build_networks(self):
        c, dev = self.cfg, self.device
        self.wm = WorldModel(self.nq, self.nm, c.wm_hidden).to(dev)
        self.pi = Policy(self.obs_policy.dim, self.obs_goal.dim, self.nm, c.enc_hidden, c.dec_hidden, c.latent_dim,
                         c.init_excitation).to(dev)  # fmt: skip
        self.critic = Critic(self.heads.critic_dim(self.nm) + self.obs_goal.dim, c.critic_hidden).to(dev)
        self.critic_targ = self.critic
        if c.critic_target_alpha > 0:
            self.critic_targ = copy.deepcopy(self.critic).requires_grad_(False)
            self.critic_targ.norm = self.critic.norm  # one input normalizer, updated with the online critic
        self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=c.lr)
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=c.lr, betas=(0.9, 0.95))
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=c.lr)

    # ------------------------------------------------------------------ analytic pieces
    def _wm_rollout(self, s, act, a):
        S, act = unflatten_body(s[:, 0], self.nq), act[:, 0]
        preds = []
        for j in range(self.cfg.wm_window):
            act, act_mean = self.muscle.step(act, a[:, j])
            S = self.wm.step(S, act_mean)
            preds.append(S)
        return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6))  # (B, k, ...)

    def _wm_err(self, S_hat, s1, mask):
        std = torch.maximum(self.wm.out_norm.std(), self.err_floor)
        e = (self.wm.state_error(S_hat, unflatten_body(s1, self.nq)) / std) ** 2  # (B, k, D)
        e = e.mean(-1)
        return (e * mask).sum() / s1.shape[0], e[:, -1]

    def _imag_step(self, S, hist, f, a_last):
        ctx = StepCtx(S, f, a_last)
        a = self.pi(self.obs_policy.assemble(hist, ctx), self.obs_goal.assemble(None, ctx))
        act1, act_mean = self.muscle.step(S[6], a)
        S1 = self.wm.step(S, act_mean) + (act1,)
        f1 = f + 1
        return S1, a, self.obs_policy.push(hist, StepCtx(S1, f1, a)), f1

    def _reward_term(self, S, f, a, a_prev):
        ctx = StepCtx(S, f, a, a_prev)
        r = sum(w * fn(self.heads, ctx, **p) for fn, w, p in self.reward_terms)
        alive = torch.ones_like(r)
        for fn, p in self.fail_terms:
            alive = alive * (1 - fn(self.heads, ctx, **p))
        return r, 1 - alive

    def _critic_input(self, S, f):
        ctx = StepCtx(S, f)
        return torch.cat((self.heads.critic_features(ctx), self.obs_goal.assemble(None, ctx)), -1)

    def value(self, x):
        """Bootstrap value: the target critic, optionally clamped to the attainable maximum."""
        v = self.critic_targ(x)
        return v.clamp(max=self.max_step_reward / (1 - self.cfg.gamma)) if self.cfg.value_clip else v

    @torch.no_grad()
    def update_critic_targ(self):
        if self.critic_targ is not self.critic:
            for pt, p in zip(self.critic_targ.parameters(), self.critic.parameters(), strict=True):
                pt.lerp_(p, 1 - self.cfg.critic_target_alpha)

    def lr_at(self, it: int) -> float:
        c = self.cfg
        if it >= c.lr_decay_iters:
            return c.lr * 0.03
        return c.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * it / c.lr_decay_iters)))

    # ------------------------------------------------------------------ updates
    def wm_update(self):
        c = self.cfg
        w = self.buffer.sample_windows(c.wm_batch, c.wm_window)
        S_hat = self.wm_rollout(w["s"], w["act"], w["a"])
        loss, e_last = self.wm_err(S_hat, w["s1"], w["mask"])
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), c.grad_clip)
        self.opt_wm.step()
        m = w["mask"][:, -1]
        return loss.detach() / c.wm_window, (e_last.detach() * m).sum() / m.sum().clamp(min=1)

    def sample_valid_starts(self, B: int):
        starts, hists, n = [], [], 0
        m, H = self.motion, self.cfg.horizon
        while n < B:
            s, h = self.buffer.sample_starts(2 * B, self.hist)
            f = s["f"]
            idx = (f + H <= m.clip_end[m.clip_id[f]]).nonzero(as_tuple=False).squeeze(-1)[: B - n]
            starts.append({k: v[idx] for k, v in s.items()}), hists.append({k: v[idx] for k, v in h.items()})
            n += len(idx)
        cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}  # noqa: E731
        return cat(starts), cat(hists)

    def policy_update(self):
        c, dev, HZ, B = self.cfg, self.device, self.cfg.horizon, self.cfg.pi_batch
        s, h = self.sample_valid_starts(B)
        S = unflatten_body(s["s"], self.nq) + (s["act"],)
        f, a_last = s["f"], s["last_a"]
        with torch.no_grad():
            hist = self.obs_policy.init_history(StepCtx(unflatten_body(h["s"], self.nq) + (h["act"],), h["f"], h["last_a"]))
        set_requires_grad(self.wm, False)
        set_requires_grad(self.critic, False)
        S0, f0 = S, f
        Ss, As, Aps, Fs = [], [], [], []
        for _ in range(HZ):
            Aps.append(a_last)
            S, a_last, hist, f = self.imag_step(S, hist, f, a_last)
            Ss.append(S), As.append(a_last), Fs.append(f)
        flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
        S_all = tuple(flat([St[i] for St in Ss]) for i in range(7))
        r, d = self.reward_term(S_all, flat(Fs), flat(As), flat(Aps))
        r, d = r.view(HZ, B), d.view(HZ, B)
        alive = torch.cumprod(1 - d, 0)  # c_{t+1}
        c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
        disc = c.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
        J = (disc * c_t * r).sum(0)
        surv = alive[-1]
        r_sum = (c_t * r).detach().sum(0)
        rews, dones = r.detach(), d.detach()
        if c.use_critic:
            J = J + (c.gamma**HZ) * surv * self.value(self.critic_input(S, f))
        loss = -J.mean()
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(self.pi.parameters(), c.grad_clip)
        nn.utils.clip_grad_value_(self.pi.parameters(), c.policy_grad_clip_value)
        if torch.isfinite(gn):
            self.opt_pi.step()
        set_requires_grad(self.wm, True)
        set_requires_grad(self.critic, True)
        # ---------------- critic: GAE(lambda) targets over the imagined transitions ----------------
        v_loss = torch.zeros((), device=dev)
        if c.use_critic:
            with torch.no_grad():
                S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(7))
                f_c = torch.cat((f0.unsqueeze(0), torch.stack(Fs)), 0).flatten()
                X = self.critic_input(S_c, f_c).view(HZ + 1, B, -1)
            self.critic.norm.update(X)
            with torch.no_grad():
                V = self.value(X)
                R = V[-1]
                targets = []
                for t in range(HZ - 1, -1, -1):
                    R = rews[t] + c.gamma * (1 - dones[t]) * ((1 - c.lam) * V[t + 1] + c.lam * R)
                    targets.append(R)
                targets = torch.stack(targets[::-1])  # (H, B)
            Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
            n = Xf.shape[0]
            mb = n // c.critic_minibatches
            for _ in range(c.critic_epochs):
                perm = torch.randperm(n, device=dev)
                for i in range(c.critic_minibatches):
                    idx = perm[i * mb : (i + 1) * mb]
                    v_loss = ((self.critic(Xf[idx]) - Yf[idx]) ** 2).mean()
                    self.opt_v.zero_grad(set_to_none=True)
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), c.grad_clip)
                    self.opt_v.step()
            self.update_critic_targ()
        return loss.detach(), v_loss.detach(), (r_sum / HZ / self.max_step_reward).mean(), gn.detach(), surv.mean().detach()

    # ------------------------------------------------------------------ simulator
    @torch.no_grad()
    def collect(self, n_steps: int, stats: dict):
        u, cmd, N = self.uenv, self.cmd, self.uenv.num_envs
        for _ in range(n_steps):
            S = cmd.read_state()
            f = cmd.f.clone()
            ep_step = u.episode_length_buf.clone()
            obs = self.obs
            self.pi.obs_norm.update(obs["policy"])
            self.pi.goal_norm.update(obs["goal"])
            logits = self.pi.logits(obs["policy"], obs["goal"])
            a = torch.sigmoid(logits + self.cfg.action_noise * torch.randn_like(logits))
            obs, rew, dones, _ = self.env.step(a)
            S1 = cmd.read_state()
            done = dones.bool()
            self.buffer.add({"s": flatten_body(S), "act": S[6], "a": a, "s1": flatten_body(S1), "r": rew, "f": f,
                             "ep_step": ep_step, "done": done})  # fmt: skip
            act_mean = self.muscle.step(S[6], a)[1]
            self.wm.in_norm.update(self.wm.inputs(S, act_mean))
            self.wm.out_norm.update(self.wm.targets(S, S1))
            self.sim_steps += N
            self.ep_ret += rew
            stats["r"].append(rew.mean() / self.max_step_reward)
            # per-term share of the normalized step reward (reward manager stores weight * value per term)
            stats["r_terms"].append(u.reward_manager._step_reward.mean(0) * self.reward_scale / self.max_step_reward)
            ids = done.nonzero(as_tuple=False).squeeze(-1)
            if len(ids) > 0:
                T = (ep_step[ids] + 1).float()
                stats["ep_len"].append(T.mean())
                stats["ep_ret"].append((self.ep_ret[ids] / (T * self.max_step_reward)).mean())
                stats["term"].append(u.reset_terminated[ids].float().sum())
                stats["done"].append(float(len(ids)))
                self.ep_ret[ids] = 0.0
                obs_dict, _ = u.reset(env_ids=ids)
                obs = TensorDict(obs_dict, batch_size=[N])
            self.obs = obs

    # ------------------------------------------------------------------ mjlab runner interface
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        del init_at_random_ep_len  # episodes already start at sampled reference frames
        c, u = self.cfg, self.uenv
        assert not u.cfg.auto_reset, "SuperMimic records terminal states: the training env needs auto_reset=False"
        self.buffer = ReplayBuffer(c.buffer_steps, u.num_envs, self.heads.body_state_dim, self.nm, self.device)
        self.ep_ret = torch.zeros(u.num_envs, device=self.device)
        self.obs = self.env.get_observations()
        logger = self._make_logger()
        stats = {"r": [], "r_terms": [], "ep_len": [], "ep_ret": [], "term": [], "done": []}
        self.collect(max(c.warmup_steps, c.wm_window + self.hist + 1), stats)
        t_start, t_log, step_acc = time.time() - self.wall_offset, time.time(), 0.0
        agg = {"wm": [], "wm8": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": []}
        start_it = self.it
        for it in range(start_it, start_it + num_learning_iterations):
            lr = self.lr_at(it)
            for o in (self.opt_wm, self.opt_pi, self.opt_v):
                for g in o.param_groups:
                    g["lr"] = lr
            step_acc += 1.0 / c.utd
            n = int(step_acc)
            step_acc -= n
            self.collect(n, stats)
            for _ in range(c.wm_batches):
                l, e8 = self.wm_update()
            agg["wm"].append(l), agg["wm8"].append(e8)
            if it >= c.wm_only_iters:
                pl, vl, ir, gn, surv = self.policy_update()
                agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn), agg["surv"].append(surv)
            self.it = it + 1
            if self.it % c.log_interval == 0:
                mean = lambda x: float(sum(float(v) for v in x) / max(len(x), 1))  # noqa: E731
                ips = c.log_interval / (time.time() - t_log)
                t_log = time.time()
                wall_h = (time.time() - t_start) / 3600
                term_frac = float(sum(float(v) for v in stats["term"]) / max(sum(float(v) for v in stats["done"]), 1))
                row = {"iteration": self.it, "wall_h": wall_h, "sim_steps": self.sim_steps, "wm_loss": mean(agg["wm"]),
                       "wm_err8": mean(agg["wm8"]), "pi_loss": mean(agg["pi"]), "v_loss": mean(agg["v"]), "imag_rew": mean(agg["ir"]),
                       "pi_gradnorm": mean(agg["gn"]), "imag_survival": mean(agg["surv"]), "sim_rew": mean(stats["r"]),
                       "ep_len": mean(stats["ep_len"]), "ep_ret_norm": mean(stats["ep_ret"]), "term_frac": term_frac,
                       "it_per_s": ips, "lr": lr,
                       "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0}  # fmt: skip
                if stats["r_terms"]:
                    terms = torch.stack(stats["r_terms"]).mean(0).tolist()
                    row.update({f"rew_{n}": float(v) for n, v in zip(self.reward_names, terms, strict=True)})
                logger(row)
                print(f"it {self.it} | {wall_h:.3f} h | wm {row['wm_loss']:.4f} e8 {row['wm_err8']:.4f} | pi {row['pi_loss']:.3f} "
                      f"v {row['v_loss']:.4f} imag_r {row['imag_rew']:.3f} surv {row['imag_survival']:.3f} | sim_r {row['sim_rew']:.3f} "
                      f"ep_len {row['ep_len']:.1f} ep_ret {row['ep_ret_norm']:.3f} term {term_frac:.2f} | {ips:.2f} it/s {row['gpu_mem_gb']:.1f} GB", flush=True)  # fmt: skip
                stats = {k: [] for k in stats}
                agg = {k: [] for k in agg}
            if self.log_dir and (self.it % c.save_interval == 0 or it + 1 == start_it + num_learning_iterations):
                self.wall_offset = time.time() - t_start
                self.save(os.path.join(self.log_dir, f"model_{self.it:06d}.pt"))

    def save(self, path: str, infos=None):
        ck = {"wm": self.wm.state_dict(), "pi": self.pi.state_dict(), "critic": self.critic.state_dict(),
              "opt_wm": self.opt_wm.state_dict(), "opt_pi": self.opt_pi.state_dict(), "opt_v": self.opt_v.state_dict(),
              "iteration": self.it, "wall_time": self.wall_offset, "sim_steps": self.sim_steps, "cfg": asdict(self.cfg),
              "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
        if self.critic_targ is not self.critic:
            ck["critic_targ"] = self.critic_targ.state_dict()
        torch.save(ck, path + ".tmp")
        os.replace(path + ".tmp", path)

    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location=None):
        ck = torch.load(path, map_location=map_location or self.device, weights_only=False)
        saved = {k: tuple(v) if isinstance(v, list) else v for k, v in ck["cfg"].items() if k in self.NET_FIELDS}
        if any(saved[k] != getattr(self.cfg, k) for k in saved):  # e.g. play/eval with the registered default cfg
            for k, v in saved.items():
                setattr(self.cfg, k, v)
            self._build_networks()
        self.pi.load_state_dict(ck["pi"], strict=strict)
        if load_cfg is None or not set(load_cfg) <= {"actor"}:
            self.wm.load_state_dict(ck["wm"], strict=strict)
            self.critic.load_state_dict(ck["critic"], strict=strict)
            if self.critic_targ is not self.critic:
                self.critic_targ.load_state_dict(ck.get("critic_targ", ck["critic"]), strict=strict)
            for k in ("opt_wm", "opt_pi", "opt_v"):
                getattr(self, k).load_state_dict(ck[k])
            self.it, self.wall_offset, self.sim_steps = ck["iteration"], ck["wall_time"], ck["sim_steps"]
        return ck

    def get_inference_policy(self, device=None):
        self.pi.eval()
        if device is not None:
            self.pi.to(device)

        @torch.no_grad()
        def policy(obs) -> torch.Tensor:
            return self.pi(obs["policy"], obs["goal"])

        return policy

    def add_git_repo_to_log(self, repo_file_path: str):
        pass

    def _make_logger(self):
        if not self.log_dir:
            return lambda row: None
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, "train_log.csv")
        new = not os.path.exists(path)
        fh = open(path, "a", newline="")
        writer, tb, wb = None, None, None
        if self.cfg.logger == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb = SummaryWriter(self.log_dir)
            except ImportError:
                print("[WARN] tensorboard is not installed; logging to train_log.csv only")
        elif self.cfg.logger == "wandb":
            import wandb

            wb = wandb.init(project=self.cfg.wandb_project, name=self.cfg.run_name or os.path.basename(self.log_dir),
                            dir=self.log_dir, config=asdict(self.cfg))  # fmt: skip

        def log(row: dict):
            nonlocal writer, new
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(row))
                if new:
                    writer.writeheader()
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
            fh.flush()
            if tb is not None:
                for k, v in row.items():
                    if k != "iteration":
                        tb.add_scalar(f"train/{k}", v, row["iteration"])
            if wb is not None:
                wb.log({f"train/{k}": v for k, v in row.items()}, step=row["iteration"])

        return log


def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)
