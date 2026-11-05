"""BlinkMimic's drop-in tier on the SuperMimic runner (docs/training.md): the Muon / Adam optimizer and its gated
step, the no-sync normalizer, the reset-statistics fixes, and a tiny end-to-end run with the whole recipe on
(EMA policy, no-sync sampling and step, policy learning-rate multiplier and its decay, checkpoint round trip).

The end-to-end tests need ``data/myofullbody/walk.npz`` (docs/motion_data.md); without it they are skipped.
"""

import copy
import csv
import os
from dataclasses import asdict

import pytest
import torch

from muscle_supermimic.rl.supermimic.networks import Policy, RunningNorm
from muscle_supermimic.rl.supermimic.optim import MuonAdam, gated_step

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
TASK, CLIP = "Mjlab-SuperMimic-Flat-MyoFullBody", os.path.join(DATA, "myofullbody", "walk.npz")


def _policy(seed: int = 0) -> Policy:
    torch.manual_seed(seed)
    return Policy(20, 30, 5, enc_hidden=(64, 32, 16), dec_hidden=(64, 32, 16))


def _snapshot(m, opt) -> list[torch.Tensor]:
    """Parameters and every optimizer state tensor."""
    t = [p.detach().clone() for p in m.parameters()]
    for sub in (opt.muon, opt.adam) if isinstance(opt, MuonAdam) else (opt,):
        for st in sub.state.values():
            t += [v.detach().clone() for v in st.values() if torch.is_tensor(v)]
    return t


def _set_grads(m, seed: int):
    g = torch.Generator().manual_seed(seed)
    for p in m.parameters():
        p.grad = torch.randn(p.shape, generator=g)


def test_muon_adam_puts_exactly_the_hidden_matrices_on_muon():
    pi = _policy()
    opt = MuonAdam(pi, [pi.enc, pi.dec], lr=1e-3, fused=False)
    muon = [p for g in opt.muon.param_groups for p in g["params"]]
    adam = [p for g in opt.adam.param_groups for p in g["params"]]
    assert len(muon) == 4 and all(p.dim() == 2 for p in muon)  # two hidden Linear weights per MLP
    assert len(muon) + len(adam) == len(list(pi.parameters()))
    first_last = {id(pi.enc[0].weight), id(pi.enc[-1].weight), id(pi.dec[0].weight), id(pi.dec[-1].weight)}
    assert first_last <= {id(p) for p in adam}


def test_muon_adam_moves_every_parameter_and_round_trips_its_state():
    pi = _policy()
    opt = MuonAdam(pi, [pi.enc, pi.dec], lr=1e-2, fused=False)
    before = [p.detach().clone() for p in pi.parameters()]
    for i in range(3):
        _set_grads(pi, i)
        opt.step()
    assert all(not torch.equal(p, b) for p, b in zip(pi.parameters(), before))
    pi2 = _policy(1)
    opt2 = MuonAdam(pi2, [pi2.enc, pi2.dec], lr=1e-2, fused=False)
    opt2.load_state_dict(opt.state_dict())
    for g in opt2.param_groups:
        g["lr"] = 5e-3
    assert all(g["lr"] == 5e-3 for g in opt2.param_groups)  # the property sees the loaded groups


@pytest.mark.parametrize("kind", ["muon", "adam"])
def test_gated_step_is_the_plain_step_or_a_no_op(kind):
    a = _policy()
    b = copy.deepcopy(a)

    def make(m):
        if kind == "muon":
            return MuonAdam(m, [m.enc, m.dec], lr=1e-3, fused=True)
        return torch.optim.Adam(m.parameters(), lr=1e-3, fused=True)

    oa, ob = make(a), make(b)
    for i in range(3):
        _set_grads(a, i), _set_grads(b, i)
        oa.step(), gated_step(ob, torch.tensor(True))
    for x, y in zip(_snapshot(a, oa), _snapshot(b, ob), strict=True):
        assert torch.equal(x, y)  # ok=True is the plain step, bitwise
    before = _snapshot(b, ob)
    _set_grads(b, 99)
    gated_step(ob, torch.tensor(False))
    for x, y in zip(before, _snapshot(b, ob), strict=True):
        assert torch.equal(x, y)  # ok=False changes neither the parameters nor any optimizer state (step counts included)


def test_running_norm_no_sync_matches_the_host_side_freeze():
    torch.manual_seed(0)
    a, b = RunningNorm(5, max_count=25), RunningNorm(5, max_count=25)
    b.no_sync = True
    for _ in range(6):  # frozen from the 5th batch on (count 32 >= 25)
        x = torch.randn(8, 5) * 3 + 1
        a.update(x), b.update(x)
        torch.testing.assert_close(a.mean, b.mean, atol=0, rtol=0)
        torch.testing.assert_close(a.var, b.var, atol=0, rtol=0)
        assert float(a.count) == float(b.count)
    assert float(a.count) == 32
    torch.testing.assert_close(a(x), b(x), atol=0, rtol=0)


def test_bin_ema_averages_the_episodes_of_a_bin():
    from muscle_supermimic.tasks.tracking.mdp.commands import bin_ema

    stat = torch.tensor([1.0, 2.0, 3.0])
    bins = torch.tensor([0, 0, 2, 2, 2])
    x = torch.tensor([3.0, 5.0, 0.0, 6.0, 100.0])
    wt = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])  # the last sample is masked out
    out = bin_ema(stat, bins, x, wt, 0.8)
    torch.testing.assert_close(out, torch.tensor([0.8 * 1 + 0.2 * 4, 2.0, 0.8 * 3 + 0.2 * 3]))


# ---------------------------------------------------------------- end to end (CPU, 2 envs, tiny networks)


@pytest.fixture(scope="module")
def setup():
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    if not os.path.exists(CLIP):
        pytest.skip(f"no test clip at {CLIP}")
    torch.manual_seed(0)
    cfg = load_env_cfg(TASK)
    cmd = cfg.commands["motion"]
    cmd.motion_file, cmd.sampling_mode, cmd.bin_fix, cmd.fail_reset = CLIP, "start", True, 0.3
    cfg.scene.num_envs = 2
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cpu"))
    rl = asdict(load_rl_cfg(TASK))
    rl.update(wm_hidden=(64, 64, 64), enc_hidden=(64, 32, 16), dec_hidden=(64, 32, 16), critic_hidden=(32,), buffer_steps=40,
              warmup_steps=20, wm_batch=16, wm_batches=2, pi_batch=8, horizon=4, log_interval=1, save_interval=1000,
              # the drop-in recipe (scripts/train_blink.sh), scaled down
              pi_ema=0.9, pi_opt="muon", wm_opt="muon", fused_adam=True, no_sync=True, wm_only_iters=0, pi_lr_mult=4.0,
              lr_decay_ret=2.0, lr_decay_by=2, lr_decay_len=2, empty_cache=False, wm_structured=True, wm_mirror=0.5,
              eprobe_share=0.15, eprobe_start=1, eprobe_every=2, eprobe_tile_s=0.5, eprobe_lead_s=(0.1, 0.2), eprobe_win_s=2.0)  # fmt: skip
    runner = load_runner_cls(TASK)(env, rl, device="cpu")
    return env, runner


def test_reset_probs_move_fail_reset_before_the_failure_bins(setup):
    _, runner = setup
    cmd, nb = runner.cmd, runner.motion.num_bins
    assert nb >= 3
    cmd.set_focus(None, 0.0)
    cmd.bin_G.zero_()  # the return-based mixture is then exactly uniform (bin_w_avg stays at its uniform start)
    cmd.bin_fail.zero_()
    b = nb - 1
    cmd.bin_fail[b] = 0.5
    probs = cmd.reset_probs()
    expect = torch.full((nb,), 0.7 / nb)
    expect[cmd.bin_prev[b]] += 0.3
    torch.testing.assert_close(probs, expect, atol=1e-6, rtol=0)
    assert cmd.bin_prev[0] == 0  # a clip's first bin has no predecessor
    cmd.bin_fail.zero_()
    torch.testing.assert_close(cmd.reset_probs(), torch.full((nb,), 1.0 / nb), atol=1e-6, rtol=0)  # no failures: unchanged
    # the focus distribution of the evaluation-guided resets mixes in with its share
    focus = torch.zeros(nb)
    focus[1] = 1.0
    cmd.set_focus(focus, 0.15)
    torch.testing.assert_close(cmd.reset_probs(), 0.85 / nb + 0.15 * focus, atol=1e-6, rtol=0)
    cmd.set_focus(None, 0.0)


def test_focus_spreads_failures_over_the_window_before_them(setup):
    _, runner = setup
    m, nb = runner.motion, runner.motion.num_bins
    win = round(runner.cfg.eprobe_win_s / runner.cmd.cfg.bin_seconds)
    assert runner.focus_from_failures(torch.zeros(nb)) is None
    fail = torch.zeros(nb)
    b = nb - 1  # a failure in the last bin of the (only) clip
    fail[b] = 2.0
    q = runner.focus_from_failures(fail)
    first = max(int((m.bin_clip == m.bin_clip[b]).nonzero().min()), b - win)
    expect = torch.zeros(nb)
    expect[first : b + 1] = 1.0 / (b + 1 - first)
    torch.testing.assert_close(q, expect, atol=1e-6, rtol=0)
    assert abs(float(q.sum()) - 1.0) < 1e-6


def test_runner_with_the_recipe_learns_and_round_trips(setup, tmp_path):
    env, runner = setup
    u, c = env.unwrapped, runner.cfg
    runner.log_dir = str(tmp_path)
    runner.learn(num_learning_iterations=4)
    assert runner.it == 4 and runner._ema_n == 4 and int(runner.pi_avg.n_averaged) == 4 and int(runner.nonfinite) == 0
    # the EMA average lags the live policy, and is what the inference policy runs
    live, avg = runner.pi.state_dict(), runner.pi_avg.module.state_dict()
    assert any(not torch.equal(live[k], avg[k]) for k in live)
    obs = env.get_observations()
    torch.testing.assert_close(runner.get_inference_policy()(obs), runner.pi_avg.module(obs["policy"], obs["goal"]), atol=0, rtol=0)
    # the policy learning-rate multiplier: x4 up to the deadline (the log of iteration 2 triggers it), then, from the
    # iteration after the trigger, a ramp to x1 over 2 iterations: the multipliers used by iterations 1-4
    with open(os.path.join(tmp_path, "train_log.csv")) as fh:
        rows = list(csv.DictReader(fh))
    assert runner._decay_at == 2
    mults = [float(r["pi_mult"]) for r in rows]
    assert mults[:3] == [4.0, 4.0, 4.0] and mults[3] == 2.5
    assert all(abs(float(r["pi_lr"]) - float(r["lr"]) * float(r["pi_mult"])) < 1e-12 for r in rows)
    assert all(r["nonfinite"] == "0" for r in rows)
    # no-sync sampling: pi_batch valid starts
    s, h = runner.sample_valid_starts(c.pi_batch)
    m = runner.motion
    assert s["f"].shape == (c.pi_batch,) and (s["f"] + c.horizon <= m.clip_end[m.clip_id[s["f"]]]).all()
    assert h["s"].shape[0] == c.pi_batch
    # an episode end feeds the reset statistics (bin_fix and fail_reset paths)
    cmd = runner.cmd
    bin_G0, bin_fail0 = cmd.bin_G.clone(), cmd.bin_fail.clone()
    u.reset(env_ids=torch.tensor([0, 1]))  # both envs ended, in the same bin: averaged, one EMA step
    assert not torch.equal(cmd.bin_G, bin_G0)
    assert (cmd.bin_G != bin_G0).sum() == 1 and (cmd.bin_fail <= bin_fail0 + 1e-6).all()  # no termination: the rate decays
    # checkpoint: "pi" is the deployed (EMA) policy, "pi_live" the live one; an actor-only load reads "pi" into it
    ck_path = os.path.join(tmp_path, "model_000004.pt")
    ck = torch.load(ck_path, weights_only=False)
    for k in avg:
        torch.testing.assert_close(ck["pi"][k], avg[k], atol=0, rtol=0)
        torch.testing.assert_close(ck["pi_live"][k], live[k], atol=0, rtol=0)
    assert ck["ctrl"]["decay_at"] == 2 and ck["ctrl"]["nonfinite"] == 0
    # the evaluation-guided resets swept once (iteration 2): the probe env exists, with one world per tile of the clip,
    # its resets land on the frames asked for, and the sweep's failure fraction, the counts and the focus were saved
    m, probe = runner.motion, runner.probe
    assert probe is not None and probe["P"] == len(probe["tiles"]) and probe["cmd"] is not runner.cmd
    assert int(m.clip_len[0]) // probe["tile"] <= probe["P"] <= int(m.clip_len[0]) // probe["tile"] + 1
    assert 0.0 <= runner.probe_fail <= 1.0 and float(runner.ep_fail.sum()) >= 0
    assert ck["ctrl"]["probe_fail"] == runner.probe_fail and torch.equal(ck["ctrl"]["ep_fail"], runner.ep_fail)
    pcmd, pu = probe["cmd"], probe["env"].unwrapped
    f0 = pcmd.fixed_frames
    assert f0.shape == (probe["P"],) and (f0 <= probe["tiles"]).all() and (probe["tiles"] - f0 <= 20).all()
    pu.reset(env_ids=torch.arange(probe["P"]))
    assert torch.equal(pcmd.f, f0)
    torch.testing.assert_close(pcmd.read_state()[0], m.root_p[f0], atol=1e-6, rtol=0)
    for _ in range(3):  # the frame cap holds: f never passes a probe's stop
        probe["env"].step(torch.full((probe["P"], runner.nm), 0.1))
    assert (pcmd.f <= pcmd.f_cap).all()
    if runner.cmd.focus_q is not None:
        assert abs(float(runner.cmd.focus_q.sum()) - 1.0) < 1e-5 and runner.cmd.focus_share == 0.15
        assert torch.equal(ck["ctrl"]["focus_q"], runner.cmd.focus_q)
    with torch.no_grad():
        for p in runner.pi_avg.module.parameters():
            p.add_(1.0)
    runner.load(ck_path, load_cfg={"actor": True})
    for k, v in runner.pi_avg.module.state_dict().items():
        torch.testing.assert_close(v, avg[k], atol=0, rtol=0)
    # a full load restores the EMA state, the trigger, the counters and the focus, and training continues
    runner._decay_at, runner._ema_n = None, 0
    focus_q, focus_share = runner.cmd.focus_q, runner.cmd.focus_share
    runner.cmd.set_focus(None, 0.0)
    runner.ep_fail.zero_()
    runner.load(ck_path)
    assert runner._decay_at == 2 and runner._ema_n == 4 and runner.it == 4
    assert torch.equal(runner.ep_fail, ck["ctrl"]["ep_fail"]) and runner.cmd.focus_share == focus_share
    assert (focus_q is None and runner.cmd.focus_q is None) or torch.equal(runner.cmd.focus_q, focus_q)
    runner.learn(num_learning_iterations=2)
    assert runner.it == 6 and runner._ema_n == 6
    # the structured world model and the mirror were on: the increments' normalizer differs from the targets' one, and
    # mirrored windows keep their shapes, mirror back to themselves and leave the episode-end mask alone
    assert runner.cfg.wm_structured and runner.mirror is not None and float(runner.wm.inc_norm.count) > 0
    assert not torch.allclose(runner.wm.inc_norm.mean, runner.wm.out_norm.mean)
    w = runner.buffer.sample_windows(6, c.wm_window)
    wm_ = runner._mirror_windows(w, 3)
    assert all(wm_[k].shape == w[k].shape for k in w) and torch.equal(wm_["mask"], w["mask"])
    assert not torch.allclose(wm_["s"][:3], w["s"][:3]) and torch.equal(wm_["s"][3:], w["s"][3:])
    back = runner._mirror_windows(wm_, 3)
    for k in ("s", "s1", "act", "a", "touch1"):
        torch.testing.assert_close(back[k], w[k], atol=1e-5, rtol=1e-5)
    # a checkpoint of the code before inc_norm loads: out_norm stands in for it
    old = {**ck, "wm": {k: v for k, v in ck["wm"].items() if not k.startswith("inc_norm.")}}
    torch.save(old, os.path.join(tmp_path, "old.pt"))
    runner.load(os.path.join(tmp_path, "old.pt"))
    torch.testing.assert_close(runner.wm.inc_norm.mean, runner.wm.out_norm.mean, atol=0, rtol=0)


def test_beta_exploration_noise_is_unbiased_and_stays_in_the_box():
    from muscle_supermimic.rl.supermimic.networks import explore

    torch.manual_seed(0)
    a = torch.tensor([0.0, 1e-4, 0.02, 0.2, 0.5, 0.9, 1.0]).repeat(40000, 1)
    x = explore(a, 0.1, "beta")
    assert torch.isfinite(x).all() and (x >= 0).all() and (x <= 1).all()
    k = 1 / (4 * 0.1**2) - 1  # 24: standard deviation 0.1 at 0.5
    torch.testing.assert_close(x.mean(0), a[0], atol=2e-3, rtol=0)  # mean-preserving, also next to the bounds
    torch.testing.assert_close(x.std(0), (a[0] * (1 - a[0]) / (k + 1)).sqrt(), atol=2e-3, rtol=0.02)
    assert torch.equal(x[:, 0], a[:, 0]) and torch.equal(x[:, -1], a[:, -1])  # no noise at the bounds themselves
    # the clipped Gaussian it replaces is biased upwards near 0
    g = explore(a, 0.1, "gaussian")
    assert float(g[:, 2].mean()) > 2 * 0.02


def test_reflected_exploration_noise_folds_the_gaussian_into_the_box():
    import math

    from muscle_supermimic.rl.supermimic.networks import explore

    torch.manual_seed(0)
    a = torch.tensor([0.0, 0.02, 0.5, 0.95, 1.0]).repeat(200000, 1)
    x = explore(a, 0.1, "reflect")
    assert (x >= 0).all() and (x <= 1).all()
    assert float((x == 0).float().mean()) < 1e-4 and float((x == 1).float().mean()) < 1e-4  # nothing piled on the bounds
    # next to a bound: the folded normal |N(m, s^2)|, mean s sqrt(2/pi) exp(-m^2 / 2 s^2) + m erf(m / s sqrt(2))
    s = 0.1

    def folded(m):
        return s * math.sqrt(2 / math.pi) * math.exp(-(m**2) / (2 * s**2)) + m * math.erf(m / (s * math.sqrt(2)))

    assert abs(float(x[:, 0].mean()) - folded(0.0)) < 1e-3  # a = 0: the half-normal
    assert abs(float(x[:, 1].mean()) - folded(0.02)) < 1e-3  # 0.081
    assert abs(float(x[:, 3].mean()) - (1 - folded(0.05))) < 1e-3  # the upper bound mirrors the lower
    assert abs(float(x[:, 4].mean()) - (1 - folded(0.0))) < 1e-3
    # in the middle it is the plain Gaussian
    assert abs(float(x[:, 2].mean()) - 0.5) < 1e-3 and abs(float(x[:, 2].std()) - 0.1) < 1e-3
    # far outside the box (a large sigma) the triangle wave still lands inside
    y = explore(torch.full((1000,), 0.5), 3.0, "reflect")
    assert (y >= 0).all() and (y <= 1).all() and float(y.std()) > 0.2


def test_sampling_from_a_snapshot_never_reaches_the_rows_being_written():
    """overlap_collect: sampling reads the write head and fill level of the last snapshot, capped so that the next
    `reserve` rows (overwritten by the concurrent collection) are never read; writes after the snapshot do not move it."""
    from muscle_supermimic.rl.supermimic.buffer import ReplayBuffer

    torch.manual_seed(0)
    T, N, reserve = 10, 3, 3
    buf = ReplayBuffer(T, N, body_dim=4, num_muscles=2, num_touch=1, device="cpu")
    tr = {k: torch.zeros((N,) + shape, dtype=dt) for k, (shape, dt) in buf.fields.items()}
    for _ in range(12):  # full and wrapped: the oldest rows are the next ones written
        buf.add(tr)
    buf.use_device_ptr()
    buf.use_sample_snapshot(reserve)
    rows = []
    row = buf._row
    buf._row = lambda age: rows.append(row(age).flatten()) or row(age)

    def sampled():
        rows.clear()
        buf.sample_windows(400, 2)
        buf.sample_starts(400, 1)
        return set(torch.cat(rows).tolist())

    ptr = buf.ptr
    in_flight = {(ptr + i) % T for i in range(reserve)}
    first = sampled()
    assert first == set(range(T)) - in_flight  # everything else, the newest row included
    for _ in range(reserve):  # the concurrent collection writes the reserved rows
        buf.add_device(tr)
    assert sampled() == first  # sampling still reads the snapshot
    buf.snapshot()
    assert sampled() == set(range(T)) - {(ptr + reserve + i) % T for i in range(reserve)}
