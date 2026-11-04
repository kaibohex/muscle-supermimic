"""The CUDA graphs of the updates (``cuda_graph``, BlinkMimic): the capture wrapper replays exactly what it captured,
and a tiny training run on the GPU captures both updates after their warm-up calls, keeps its learning-rate tensors
on the schedule, advances the policy EMA inside the replays and round-trips a checkpoint. Needs a CUDA device and
``data/myofullbody/walk.npz``."""

import copy
import csv
import os
from dataclasses import asdict

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
TASK, CLIP = "Mjlab-SuperMimic-Flat-MyoFullBody-No-Muscle-Obs", os.path.join(DATA, "myofullbody", "walk.npz")


def test_graphed_replays_the_captured_update():
    from muscle_supermimic.rl.supermimic.runner import Graphed, set_lr

    torch.manual_seed(0)
    dev = "cuda:0"
    x = torch.randn(64, 8, device=dev)
    nets = [nn.Linear(8, 4).to(dev) for _ in range(2)]
    nets[1].load_state_dict(nets[0].state_dict())
    opts = [torch.optim.Adam(n.parameters(), lr=torch.tensor(1e-2, device=dev), fused=True, capturable=True) for n in nets]

    def make_step(net, opt):
        def step():
            opt.zero_grad(set_to_none=True)
            loss = (net(x) ** 2).mean()
            loss.backward()
            opt.step()
            return (loss.detach(),)

        return step

    eager, graphed = make_step(nets[0], opts[0]), Graphed(make_step(nets[1], opts[1]), lambda: True)
    losses = []
    for i in range(10):
        set_lr(opts[0], 1e-2 * (1 - 0.05 * i)), set_lr(opts[1], 1e-2 * (1 - 0.05 * i))
        (le,), (lg,) = eager(), graphed()
        losses.append((float(le), float(lg)))
    assert graphed.graph is not None and graphed.calls == 10  # captured on the 4th call, replayed since
    for p, q in zip(nets[0].parameters(), nets[1].parameters(), strict=True):
        torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-5)
    assert all(abs(a - b) < 1e-5 for a, b in losses) and losses[-1][0] < losses[0][0]


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
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
    rl = asdict(load_rl_cfg(TASK))
    rl.update(wm_hidden=(64, 64, 64), enc_hidden=(64, 32, 16), dec_hidden=(64, 32, 16), critic_hidden=(32,), buffer_steps=40,
              warmup_steps=20, wm_batch=16, wm_batches=2, pi_batch=8, horizon=4, log_interval=1, save_interval=1000, compile=False,
              pi_ema=0.9, pi_opt="muon", wm_opt="muon", fused_adam=True, no_sync=True, wm_only_iters=0, empty_cache=False,
              wm_structured=True, wm_mirror=0.5, value_clip=True, cuda_graph=True)  # fmt: skip
    runner = load_runner_cls(TASK)(env, rl, device="cuda:0")
    return env, runner


def test_updates_run_as_cuda_graphs(setup, tmp_path):
    _, runner = setup
    runner.log_dir = str(tmp_path)
    runner.learn(num_learning_iterations=6)
    g = runner.graphs
    # captured after 3 warm-up calls each; the policy's first update (the EMA's copy) ran eagerly before its wrapper
    # counted calls, so the policy graph saw 5 of the 6 updates
    assert g["wm"].graph is not None and g["policy"].graph is not None
    assert g["wm"].calls == 12 and g["policy"].calls == 5
    with open(os.path.join(tmp_path, "train_log.csv")) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 6 and all(r["nonfinite"] == "0" for r in rows)
    for k in ("wm_loss", "wm_err8", "pi_loss", "v_loss", "imag_rew", "pi_gradnorm"):
        assert all(abs(float(r[k])) < 1e6 for r in rows), k
    assert int(runner.pi_avg.n_averaged) == 6  # the EMA step ran inside the replays
    assert runner._ema_n == 5  # the Python side only: the first update, the 3 warm-ups and the capture itself
    lr_v = runner.opt_v.param_groups[0]["lr"]
    assert torch.is_tensor(lr_v) and abs(float(lr_v) - runner.lr_at(5)) < 1e-9  # a float32 device tensor
    assert all(abs(float(r["pi_lr"]) - float(r["lr"])) < 1e-12 for r in rows)
    # a checkpoint round trip keeps the device learning-rate tensors the graphs read, and training continues in them
    ck = os.path.join(tmp_path, "model_000006.pt")
    assert os.path.exists(ck)
    pi_before = copy.deepcopy(runner.pi.state_dict())
    runner.load(ck)
    assert runner.opt_v.param_groups[0]["lr"] is lr_v and runner.opt_pi.param_groups[0]["lr"] is runner.opt_pi.param_groups[0]["lr"]
    assert runner.graphs is None  # the loaded optimizer states are new tensors: the captured updates are dropped
    runner.learn(num_learning_iterations=2)  # recaptures (eager warm-ups first)
    assert runner.it == 8 and int(runner.pi_avg.n_averaged) == 8 and runner.graphs["policy"].calls == 2
    assert any(not torch.equal(v, pi_before[k]) for k, v in runner.pi.state_dict().items())


def test_collection_as_a_cuda_graph_and_the_wall_clock_budget(tmp_path):
    """collect_graph: a tiny run collects through the captured step (episode ends and resets inside the graph), logs
    the episode statistics, keeps the host and device write heads together, and max_hours stops training with a final
    checkpoint."""
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    if not os.path.exists(CLIP):
        pytest.skip(f"no test clip at {CLIP}")
    torch.manual_seed(0)
    cfg = load_env_cfg(TASK)
    cmd = cfg.commands["motion"]
    cmd.motion_file, cmd.sampling_mode, cmd.bin_fix, cmd.fail_reset = CLIP, "return", True, 0.3
    cfg.scene.num_envs = 8
    cfg.episode_length_s = 0.3  # episode ends within the run
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
    rl = asdict(load_rl_cfg(TASK))
    rl.update(wm_hidden=(64, 64, 64), enc_hidden=(64, 32, 16), dec_hidden=(64, 32, 16), critic_hidden=(32,), buffer_steps=40,
              warmup_steps=20, wm_batch=16, wm_batches=2, pi_batch=8, horizon=4, log_interval=2, save_interval=1000, compile=False,
              pi_ema=0.9, pi_opt="muon", wm_opt="muon", fused_adam=True, no_sync=True, wm_only_iters=0, empty_cache=False,
              wm_structured=True, wm_mirror=0.5, value_clip=True, cuda_graph=True, collect_graph=True)  # fmt: skip
    runner = load_runner_cls(TASK)(env, rl, device="cuda:0")
    runner.log_dir = str(tmp_path)
    runner.learn(num_learning_iterations=10)
    col, buf = runner.collector, runner.buffer
    assert col.graph is not None and col.calls == 20 + 25  # the warm-up collection, then 2.5 steps per iteration
    assert buf.ptr == int(buf.ptr_t) and buf.size == int(buf.size_t) == min(col.calls, buf.T)
    assert runner.sim_steps == col.calls * 8
    with open(os.path.join(tmp_path, "train_log.csv")) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 5 and all(r["nonfinite"] == "0" for r in rows)
    for k in ("sim_rew", "ep_len", "ep_ret_norm", "wm_loss", "pi_loss"):
        assert all(abs(float(r[k])) < 1e6 for r in rows), k
    assert any(float(r["ep_len"]) > 0 for r in rows) and all(0 <= float(r["term_frac"]) <= 1 for r in rows)
    assert float(runner.cmd.bin_G.abs().sum()) > 0  # episode ends reached the bin statistics
    # the wall-clock budget: the earlier run's clock already exceeds it, so the next learn() stops after one iteration
    runner.cfg.max_hours = 1e-9
    runner.learn(num_learning_iterations=100)
    assert runner.it == 11 and os.path.exists(os.path.join(tmp_path, "model_000011.pt"))
