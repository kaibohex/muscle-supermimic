"""The mjlab task and the SuperMimic runner on CPU (MuJoCo-Warp's CPU backend), 2 envs.

Checks that the simulator side (mjlab managers) and the imagination side (analytic heads) agree:
reset states, muscle activations, rewards, the policy observation history and the goal. Then runs a
few tiny training iterations end to end and round-trips a checkpoint.

Needs a clip: ``data/motions/walk.npz`` (see README), otherwise the module is skipped.
"""

import os
from dataclasses import asdict

import pytest
import torch

MOTION = os.path.join(os.path.dirname(__file__), "..", "data", "motions", "walk.npz")
TASKS = ["Mjlab-SuperMimic-Flat-MS-Human-700", "Mjlab-SuperMimic-Flat-MS-Human-700-Locked-Hands"]
pytestmark = pytest.mark.skipif(not os.path.exists(MOTION), reason="no test clip at data/motions/walk.npz")


@pytest.fixture(scope="module", params=TASKS)
def setup(request):
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    TASK = request.param
    torch.manual_seed(0)
    cfg = load_env_cfg(TASK)
    cfg.commands["motion"].motion_file = MOTION
    cfg.commands["motion"].sampling_mode = "start"  # deterministic resets, far from the clip end
    cfg.scene.num_envs = 2
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cpu"))
    rl = asdict(load_rl_cfg(TASK))
    rl.update(wm_hidden=(64, 64), enc_hidden=(64,), dec_hidden=(64,), critic_hidden=(32,), buffer_steps=40, warmup_steps=20,
              wm_batch=16, wm_batches=2, pi_batch=8, horizon=4, wm_only_iters=2, log_interval=2, save_interval=1000)  # fmt: skip
    runner = load_runner_cls(TASK)(env, rl, device="cpu")
    return env, runner


def test_reset_writes_reference_state(setup):
    _, runner = setup
    cmd = runner.cmd
    S = cmd.read_state()
    p, R, _, w, q, qd = cmd.motion.ref_state(cmd.f)
    torch.testing.assert_close(S[0], p, atol=1e-6, rtol=0)
    torch.testing.assert_close(S[1], R, atol=1e-6, rtol=0)
    torch.testing.assert_close(S[3], w, atol=1e-5, rtol=0)
    torch.testing.assert_close(S[4], q.clamp(cmd.kin.q_lo, cmd.kin.q_hi), atol=1e-6, rtol=0)
    torch.testing.assert_close(S[5], qd, atol=1e-6, rtol=0)
    assert S[6].abs().max() == cmd.cfg.init_activation
    assert S[7].shape == (2, 4) and (S[7] >= 0).all()


def test_sim_matches_analytic_heads(setup):
    from muscle_supermimic.tasks.tracking.mdp.heads import StepCtx

    env, runner = setup
    u, cmd, m = env.unwrapped, runner.cmd, runner.motion
    env.unwrapped.reset()
    gen = torch.Generator().manual_seed(0)
    for _ in range(12):
        S0, a_prev = cmd.read_state(), u.action_manager.action.clone()
        a = torch.rand(2, cmd.info.num_muscles, generator=gen) * 0.3
        obs, rew, dones, _ = env.step(a)
        assert not dones.any()
        S1 = cmd.read_state()
        # muscle activation: simulator == analytic integration
        torch.testing.assert_close(S1[6], cmd.muscle.step(S0[6], a)[0], atol=1e-6, rtol=0)
        # reward: reward manager == the runner's imagined reward on the same transition
        r, _ = runner._reward_term(S1, cmd.f, a, a_prev)
        torch.testing.assert_close(r, rew, atol=1e-6, rtol=1e-5)
        # failure terminations: termination manager == hard analytic terms
        tm = u.termination_manager
        for name in tm.active_terms:
            tc = tm.get_term_cfg(name)
            if not tc.time_out:
                p = {k: v for k, v in tc.params.items() if k not in ("command_name", "soft_width")}
                assert torch.equal(tc.func.analytic(runner.heads, StepCtx(S1, cmd.f), **p), tm.get_term(name)), name
        # observations: the manager's groups == the analytic groups on the same state
        ctx = cmd.obs_ctx()
        torch.testing.assert_close(runner.obs_policy.assemble(None, ctx), obs["policy"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(runner.obs_goal.assemble(None, ctx), obs["goal"], atol=1e-6, rtol=0)
    assert S1[7].sum() > 0  # the feet touch the ground
    assert m.num_frames > 0


def test_deviation_terminations(setup):
    """MuscleMimic's deviation checks: zero on the reference, each fires on its own perturbation."""
    import math

    from muscle_supermimic.tasks.tracking.mdp import heads as H
    from muscle_supermimic.tasks.tracking.mdp.heads import StepCtx

    _, runner = setup
    cmd, h = runner.cmd, runner.heads
    assert h.kin.body_names[0] == "pelvis" and h.kin.tracked_idx[0] == 0
    f = torch.arange(0, 200, 25)
    p, R, v, w, q, qd = cmd.motion.ref_state(f)
    act, touch = torch.zeros(len(f), cmd.info.num_muscles), torch.zeros(len(f), 4)
    terms = (H.bad_body_pos_dev, H.bad_root_pos_dev, H.bad_root_ori_dev)

    def fired(p, R, threshold):  # fraction of frames on which each term fires
        ctx = StepCtx((p, R, v, w, q, qd, act, touch), f)
        return [fn(h, ctx, threshold=threshold).float().mean().item() for fn in terms]

    assert fired(p, R, 1e-3) == [0.0, 0.0, 0.0]  # robot on the reference
    shift = torch.tensor([0.0, 1.2, 0.0])
    assert fired(p + shift, R, 1.0) == [0.0, 1.0, 0.0]  # 1.2 m sideways drift, same pose
    c, s = math.cos(1.2), math.sin(1.2)
    Rz = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    assert fired(p, Rz @ R, 1.0)[1:] == [0.0, 1.0]  # 1.2 rad heading error
    # soft versions: one half at the threshold, finite gradients on the reference
    soft = H.bad_root_pos_dev(h, StepCtx((p + torch.tensor([0.0, 1.0, 0.0]), R, v, w, q, qd, act, touch), f), 1.0, 0.05)
    torch.testing.assert_close(soft, torch.full_like(soft, 0.5), atol=1e-4, rtol=0)
    pg, Rg = p.clone().requires_grad_(), R.clone().requires_grad_()
    ctx = StepCtx((pg, Rg, v, w, q, qd, act, touch), f)
    sum(fn(h, ctx, threshold=1.0, soft_width=0.05).sum() for fn in terms).backward()
    assert torch.isfinite(pg.grad).all() and torch.isfinite(Rg.grad).all()


def test_mimic_reward_on_reference(setup):
    """Every MimicReward component is at its maximum on the reference state, the critic's errors vanish."""
    from muscle_supermimic.tasks.tracking.mdp.heads import StepCtx

    _, runner = setup
    cmd, h = runner.cmd, runner.heads
    f = torch.arange(0, 300, 30)
    S = cmd.motion.ref_state(f) + (torch.zeros(len(f), cmd.info.num_muscles), torch.zeros(len(f), 4))
    ctx = StepCtx(S, f, torch.zeros(len(f), cmd.info.num_muscles))
    for fn, _, p in runner.reward_terms:
        torch.testing.assert_close(fn(h, ctx, **p), torch.ones(len(f)), atol=1e-4, rtol=0)
    extras = h.critic_extras(ctx)
    torch.testing.assert_close(extras[:, :4], torch.tensor([0.0, 0.0, 0.0, 0.0]).expand(len(f), 4), atol=1e-5, rtol=0)
    torch.testing.assert_close(extras[:, 4], torch.ones(len(f)), atol=1e-6, rtol=0)  # cos(dyaw)
    torch.testing.assert_close(extras[:, 5:], torch.zeros_like(extras[:, 5:]), atol=1e-5, rtol=0)


def test_runner_learns_and_round_trips(setup, tmp_path):
    env, runner = setup
    runner.log_dir = str(tmp_path)
    runner.learn(num_learning_iterations=4)
    assert runner.it == 4 and runner.buffer.size > 0
    ck = os.path.join(tmp_path, "model_000004.pt")
    assert os.path.exists(ck)
    before = {k: v.clone() for k, v in runner.pi.state_dict().items()}
    runner.load(ck, load_cfg={"actor": True})
    for k, v in runner.pi.state_dict().items():
        torch.testing.assert_close(v, before[k])
    policy = runner.get_inference_policy()
    a = policy(env.get_observations())
    assert a.shape == (2, runner.nm) and torch.isfinite(a).all() and (a >= 0).all() and (a <= 1).all()


def test_target_critic_and_value_clip(setup, tmp_path):
    env, runner = setup
    runner.cfg.critic_target_alpha, runner.cfg.value_clip = 0.9, True
    runner._build_networks()
    assert runner.critic_targ is not runner.critic and runner.critic_targ.norm is runner.critic.norm
    runner.log_dir = str(tmp_path)
    runner.learn(num_learning_iterations=4)
    online, targ = runner.critic.state_dict(), runner.critic_targ.state_dict()
    assert any(not torch.equal(online[k], targ[k]) for k in online)  # the target lags
    x = 1e3 * torch.randn(16, runner.critic.norm.mean.shape[0])
    assert (runner.value(x) <= runner.max_step_reward / (1 - runner.cfg.gamma) + 1e-6).all()
    ck = os.path.join(tmp_path, f"model_{runner.it:06d}.pt")
    runner.critic_targ.load_state_dict(online)
    runner.load(ck)
    for k, v in runner.critic_targ.state_dict().items():
        torch.testing.assert_close(v, targ[k])
