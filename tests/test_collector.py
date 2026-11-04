"""The thin simulator wrapper of ``collect_graph`` (collector.py) against the eager collection through mjlab's env.

Two identical envs, the same policy and no action noise: the wrapper (eager on the CPU; three eager warm-up steps,
then one captured CUDA graph per step on a GPU) writes the same transitions, episode ends, resets, bin statistics,
normalizer statistics and simulator state as ``env.step`` + ``env.reset(ids)``. Short episodes started at
different frames and a tight root-position threshold make both failures and time-outs happen within the test. Needs ``data/myofullbody/walk.npz``.
"""

import math
import os
from dataclasses import asdict

import pytest
import torch

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
TASK, CLIP = "Mjlab-SuperMimic-Flat-MyoFullBody-No-Muscle-Obs", os.path.join(DATA, "myofullbody", "walk.npz")
DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def _make(device: str, collect_graph: bool, sampling: str = "fixed"):
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    torch.manual_seed(0)
    cfg = load_env_cfg(TASK)
    cmd = cfg.commands["motion"]
    cmd.motion_file, cmd.sampling_mode, cmd.bin_fix, cmd.fail_reset = CLIP, sampling, True, 0.3
    cfg.scene.num_envs = 6
    cfg.episode_length_s = 0.2  # time-outs after 20 steps
    cfg.terminations["root_pos_dev"].params["threshold"] = 0.05  # failures within a few steps
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=device))
    if sampling == "fixed":  # each env (re)starts at its own frame: different motions, so failures and time-outs mix
        mc = env.unwrapped.command_manager.get_term("motion")
        mc.fixed_frames = torch.arange(6, device=device) * (mc.motion.num_frames // 8)
        env.unwrapped.reset()
    rl = asdict(load_rl_cfg(TASK))
    rl.update(wm_hidden=(64, 64, 64), enc_hidden=(64, 32, 16), dec_hidden=(64, 32, 16), critic_hidden=(32,), buffer_steps=64,
              compile=False, no_sync=True, fused_adam=True, empty_cache=False, action_noise=0.0, collect_graph=collect_graph)  # fmt: skip
    return env, load_runner_cls(TASK)(env, rl, device=device)


@pytest.fixture(scope="module", params=DEVICES)
def pair(request):
    if not os.path.exists(CLIP):
        pytest.skip(f"no test clip at {CLIP}")
    eager, graphed = _make(request.param, False)[1], _make(request.param, True)[1]
    graphed.pi.load_state_dict(eager.pi.state_dict())
    for r in (eager, graphed):
        r.init_collection()
    return request.param, eager, graphed


def test_collector_matches_the_eager_env_step(pair):
    from muscle_supermimic.tasks.tracking.mdp.heads import flatten_body

    device, eager, graphed = pair
    n = 40
    stats = []
    for r in (eager, graphed):
        st = {k: [] for k in ("r", "r_terms", "ep_len", "ep_ret", "term", "done")}
        r.collect(n, st)
        stats.append(st)
    col = graphed.collector
    assert col.calls == n and (col.graph is not None) == device.startswith("cuda")
    be, bg = eager.buffer, graphed.buffer
    assert be.ptr == bg.ptr == int(bg.ptr_t) and be.size == bg.size == int(bg.size_t) == n

    # every reset the wrapper made (in its graph on a GPU) wrote the env's reference state and restarted the episode
    done, cmd = bg.data["done"][:n], graphed.cmd
    ends = int(done.sum())
    assert ends >= 6 and done[:-1].any(0).all()  # every env ended an episode at least once
    for t, e in done[:-1].nonzero().tolist():
        frame = cmd.fixed_frames[e]
        assert int(bg.data["ep_step"][t + 1, e]) == 0 and int(bg.data["f"][t + 1, e]) == int(frame)
        p, R, v, w, q, qd = cmd.motion.ref_state(frame.view(1))
        ref = flatten_body((p, R, v, w, torch.clamp(q, cmd.kin.q_lo, cmd.kin.q_hi), qd))[0]
        torch.testing.assert_close(bg.data["s"][t + 1, e], ref, atol=1e-4, rtol=1e-4)
    assert float(bg.data["act"][1:n][done[:-1]].float().abs().max()) == 0.0  # init_activation

    if device != "cpu":
        # MuJoCo-Warp's GPU reductions are not bitwise reproducible (two eager envs drift apart by ~1e-4 per step), so
        # the paths are compared over the first steps only: the eager warm-ups, the capture and five replays
        k, tol = 8, {"atol": 2e-3, "rtol": 1e-3}
        for key in ("done", "f", "ep_step"):
            assert torch.equal(be.data[key][:k], bg.data[key][:k]), key
        for key in ("s", "act", "a", "s1", "r"):
            torch.testing.assert_close(bg.data[key][:k].float(), be.data[key][:k].float(), **tol, msg=key)
        for key in ("touch", "touch1"):  # contact forces in N
            torch.testing.assert_close(bg.data[key][:k], be.data[key][:k], atol=0.5, rtol=2e-2, msg=key)
        return

    # CPU: the same computation, to rounding
    tol = {"atol": 1e-5, "rtol": 1e-5}
    for key in ("done", "f", "ep_step"):
        assert torch.equal(be.data[key][:n], bg.data[key][:n]), key
    for key in ("s", "act", "touch", "a", "s1", "touch1", "r"):
        torch.testing.assert_close(bg.data[key][:n].float(), be.data[key][:n].float(), **tol, msg=key)
    # both kinds of episode end happened: failures (tight threshold) and time-outs (20 steps)
    term = torch.tensor([float(v) for v in stats[0]["term"]]).sum()
    assert 0 < term < ends
    # the episode-end statistics: the eager path logs the steps with an end, the wrapper NaN for the others
    for key in ("ep_len", "ep_ret"):
        e = [float(v) for v in stats[0][key]]
        g = [float(v) for v in stats[1][key] if float(v) == float(v)]
        assert len(e) == len(g) and all(math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-4) for a, b in zip(e, g, strict=True)), key
    assert sum(float(v) for v in stats[0]["done"]) == sum(float(v) for v in stats[1]["done"]) == ends
    # the motion command's episode state and bin statistics, the env's episode lengths, the simulator state
    ce, cg = eager.cmd, graphed.cmd
    for key in ("f", "start_bin"):
        assert torch.equal(getattr(ce, key), getattr(cg, key)), key
    assert torch.equal(eager.uenv.episode_length_buf, graphed.uenv.episode_length_buf)
    for key in ("bin_G", "bin_fail", "ep_ret", "ep_disc_ret"):
        torch.testing.assert_close(getattr(cg, key), getattr(ce, key), **tol, msg=key)
    assert float(ce.bin_G.abs().sum()) > 0 and float(ce.bin_fail.sum()) > 0
    de, dg = eager.uenv.sim.data, graphed.uenv.sim.data
    for key in ("qpos", "qvel", "act", "ctrl"):  # ctrl: the excitation lands on the same actuators as the action term's
        torch.testing.assert_close(getattr(dg, key)._tensor, getattr(de, key)._tensor, **tol, msg=key)
    # the normalizers saw the same data
    for ne, ng in ((eager.pi.obs_norm, graphed.pi.obs_norm), (eager.wm.in_norm, graphed.wm.in_norm), (eager.wm.inc_norm, graphed.wm.inc_norm)):
        assert float(ne.count) == float(ng.count) == n * eager.uenv.num_envs
        torch.testing.assert_close(ng.mean, ne.mean, **tol)


def test_masked_reset_frames_follow_the_reset_distribution():
    """sample_frames_masked draws bins from reset_probs (inverse CDF) and frames inside them."""
    if not os.path.exists(CLIP):
        pytest.skip(f"no test clip at {CLIP}")
    _, r = _make("cpu", True, sampling="return")
    cmd, m = r.cmd, r.cmd.motion
    cmd.bin_G.copy_(torch.linspace(0, cmd.G_max, m.num_bins))  # skew the distribution towards the first bins
    p = cmd.reset_probs()
    w_avg = cmd.bin_w_avg.clone()
    counts = torch.zeros(m.num_bins)
    for _ in range(200):
        frames, bins = cmd.sample_frames_masked(torch.tensor(False))
        assert torch.equal(cmd.bin_w_avg, w_avg)  # gate off: the running average stays
        assert ((frames >= m.bin_start[bins]) & (frames < m.bin_start[bins] + m.bin_frames)).all()
        counts += torch.bincount(bins, minlength=m.num_bins).float()
    torch.testing.assert_close(counts / counts.sum(), p, atol=0.03, rtol=0)
    cmd.sample_frames_masked(torch.tensor(True))
    assert not torch.equal(cmd.bin_w_avg, w_avg)
