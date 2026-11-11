"""Runner pieces that need no environment."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from muscle_supermimic.rl.supermimic.config import SuperMimicRunnerCfg
from muscle_supermimic.rl.supermimic.runner import ClipRowGrad, SuperMimicRunner
from muscle_supermimic.tasks.tracking.mdp.commands import MotionCommand


def test_clip_row_grad_bounds_each_row_over_the_group():
    a = torch.randn(3, 4, requires_grad=True)
    b = torch.randn(3, 2, 2, requires_grad=True)
    stats = []
    a1, b1 = ClipRowGrad.apply(1.0, 2.0, stats, a, b)
    torch.testing.assert_close(a1, a, rtol=0, atol=0)
    torch.testing.assert_close(b1, b, rtol=0, atol=0)
    w = torch.tensor([10.0, 0.01, 0.3])
    ((a1.sum(-1) + b1.flatten(1).sum(-1)) * w).sum().backward()
    # row i's gradient is w_i over 8 entries: scaled norm 2 * w_i * sqrt(8) -> rows 0 and 2 exceed 1, row 1 does not
    norm = 2.0 * torch.sqrt(a.grad.square().sum(-1) + b.grad.flatten(1).square().sum(-1))
    torch.testing.assert_close(norm, torch.tensor([1.0, 2 * 0.01 * 8**0.5, 1.0]))
    torch.testing.assert_close(a.grad[1], torch.full((4,), 0.01))  # unclipped row untouched
    assert len(stats) == 1 and abs(float(stats[0]) - 2 / 3) < 1e-6


def _checkpoint_runner(num_bins: int) -> SuperMimicRunner:
    runner = object.__new__(SuperMimicRunner)
    runner.cfg = SuperMimicRunnerCfg(
        wm_hidden=(8,),
        enc_hidden=(8,),
        dec_hidden=(8,),
        latent_dim=2,
        critic_hidden=(8,),
        eprobe_share=0.3,
    )
    runner.device, runner.nq, runner.nm = "cpu", 1, 2
    runner.uenv = SimpleNamespace(step_dt=0.02, cfg=SimpleNamespace(decimation=10))
    runner.obs_policy, runner.obs_goal = SimpleNamespace(dim=3), SimpleNamespace(dim=4)
    runner.heads = SimpleNamespace(critic_extras_dim=lambda: 2)
    runner._build_networks()
    for net, opt in ((runner.wm, runner.opt_wm), (runner.pi, runner.opt_pi), (runner.critic, runner.opt_v)):
        for parameter in net.parameters():
            parameter.grad = torch.full_like(parameter, 0.1)
        opt.step()
    runner.motion = SimpleNamespace(num_bins=num_bins)
    runner.cmd = object.__new__(MotionCommand)
    runner.cmd._env = SimpleNamespace(device="cpu")
    runner.cmd.cfg = SimpleNamespace(sampling_mode="uniform", fail_reset=0.0)
    runner.cmd.motion, runner.cmd.G_max = runner.motion, 1.0
    runner.cmd.bin_G = torch.zeros(num_bins)
    runner.cmd.bin_w_avg = torch.full((num_bins,), 1.0 / num_bins)
    runner.cmd.set_focus(torch.full((num_bins,), 1.0 / num_bins), 0.3)
    runner.ep_fail = torch.arange(num_bins, dtype=torch.float32) + 1
    runner.probe_fail = 0.25
    runner.it, runner.wall_offset, runner.sim_steps = 7, 12.5, 100
    runner._decay_at = 4
    return runner


def _assert_checkpoint_training_state(runner: SuperMimicRunner, checkpoint: dict) -> None:
    for key, network in (("wm", runner.wm), ("pi_live", runner.pi), ("critic", runner.critic)):
        torch.testing.assert_close(network.state_dict(), checkpoint[key], rtol=0, atol=0)
    for key in ("opt_wm", "opt_pi", "opt_v"):
        torch.testing.assert_close(getattr(runner, key).state_dict(), checkpoint[key], rtol=0, atol=0)
    assert (runner.it, runner.wall_offset, runner.sim_steps) == (7, 12.5, 100)
    assert runner._decay_at == 4


@pytest.mark.parametrize(("failure_bins", "focus_bins"), [(3, 3), (3, 2), (2, 3), (None, 3), (3, None)])
def test_load_rebuilds_incompatible_reset_bin_statistics(
    tmp_path: Path,
    failure_bins: int | None,
    focus_bins: int | None,
) -> None:
    source = _checkpoint_runner(3)
    path = tmp_path / "old-bins.pt"
    source.save(str(path))
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["ctrl"]["ep_fail"] = None if failure_bins is None else torch.ones(failure_bins)
    checkpoint["ctrl"]["focus_q"] = None if focus_bins is None else torch.full((focus_bins,), 1.0 / focus_bins)
    torch.save(checkpoint, path)
    restored = _checkpoint_runner(2)
    if failure_bins is None:
        restored.ep_fail = None
    with pytest.warns(RuntimeWarning, match="reset-bin"):
        restored.load(str(path))
    if restored.ep_fail is not None:
        torch.testing.assert_close(restored.ep_fail, torch.zeros(2))
    assert restored.cmd.focus_q is None and restored.cmd.focus_share == 0.0
    torch.testing.assert_close(restored.cmd.reset_probs(), torch.full((2,), 0.5))
    assert torch.isnan(torch.tensor(restored.probe_fail))
    _assert_checkpoint_training_state(restored, checkpoint)


def test_load_restores_matching_reset_bin_statistics(tmp_path: Path) -> None:
    source = _checkpoint_runner(2)
    path = tmp_path / "matching-bins.pt"
    source.save(str(path))
    checkpoint = torch.load(path, weights_only=False)
    restored = _checkpoint_runner(2)
    restored.ep_fail.zero_()
    restored.cmd.set_focus(None, 0.0)
    restored.load(str(path))
    torch.testing.assert_close(restored.ep_fail, source.ep_fail, rtol=0, atol=0)
    torch.testing.assert_close(restored.cmd.focus_q, source.cmd.focus_q, rtol=0, atol=0)
    assert restored.cmd.focus_share == source.cmd.focus_share and restored.probe_fail == source.probe_fail
    _assert_checkpoint_training_state(restored, checkpoint)
