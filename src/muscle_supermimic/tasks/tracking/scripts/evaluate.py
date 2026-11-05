"""Checkpoint evaluation (SuperMimic paper Appendix F.1) on every clip of the motion library.

Each clip plays once from its first frame (one env per clip), deterministic policy; ``--action-noise``
adds training's exploration noise instead (N(0, sigma^2) on the excitations, clipped to [0, 1]). A clip
fails if its rollout triggers a training termination. Normalized return is the undiscounted reward sum
divided by its maximum attainable value.

  uv run ms-eval --checkpoint logs/rsl_rl/mshuman700_tracking/<run>/model_014000.pt \
      --motion-file data/mshuman700_kit/kit_kinesis_test.txt
  uv run ms-eval --run-dir logs/rsl_rl/mshuman700_tracking/<run> --motion-file data/mshuman700_kit/kit_kinesis_test.txt --watch

``--skip-before-min M`` (with ``--run-dir``) leaves out the checkpoints saved before M minutes of training (their
``wall_time``): on MS-Human-700 every clip still fails in the first minutes, so evaluating them only costs time.

``--batch K`` (with ``--run-dir``) evaluates K checkpoints at once in one env of K x C worlds (world k C + j plays clip
j with checkpoint k): with ~100 clips the GPU is mostly idle per step, so K checkpoints take about as long as one. A
rollout matches the one-at-a-time evaluation up to the float rounding the worlds' shared contact pools introduce (on
a 5 s clip ~1e-3 of its return; the GPU evaluation already differs that much between runs).
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch
from tensordict import TensorDict

from muscle_supermimic.tasks.tracking.mdp.motion import motion_files
from muscle_supermimic.utils.rotations import mat_to_quat


def build(task: str, motion_path: str, device: str, copies: int = 1):
    import mjlab.tasks  # noqa: F401  (registers the tasks, ours included via the entry point)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    cfg = load_env_cfg(task)
    cmd = cfg.commands["motion"]
    cmd.motion_file = motion_path
    cmd.sampling_mode = "start"
    cfg.scene.num_envs = len(motion_files(motion_path)) * copies  # world i plays clip i % C (sampling_mode "start")
    cfg.terminations.pop("time_out", None)  # every clip runs to its end
    cfg.episode_length_s = 1e9
    cfg.auto_reset = False
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=device))
    runner = load_runner_cls(task)(env, asdict(load_rl_cfg(task)), device=device)
    return env, runner


@torch.no_grad()
def evaluate(env, runner, record: str | None = None, action_noise: float = 0.0, seed: int = 0) -> dict:
    u, cmd = env.unwrapped, runner.cmd
    m, C = cmd.motion, len(cmd.motion.names)
    ids = torch.arange(C, device=u.device)
    obs = TensorDict(u.reset(env_ids=ids)[0], batch_size=[C])
    policy = runner.get_inference_policy()
    gen = torch.Generator(device=u.device).manual_seed(seed)
    ret = torch.zeros(C, device=u.device)
    done_eval = torch.zeros(C, dtype=torch.bool, device=u.device)
    failed = torch.zeros_like(done_eval)
    q_adr = torch.cat((cmd._free_q, cmd.robot.indexing.joint_q_adr.long()))
    traj = {"qpos": [], "f": [], "alive": [], "act": []}
    while not bool(done_eval.all()):
        alive = ~done_eval
        if record:
            qpos = u.sim.data.qpos[:, q_adr].clone()
            qpos[:, :3] -= u.scene.env_origins
            # clone: cmd.f is updated in place and .cpu() aliases it on a CPU device
            traj["qpos"].append(qpos.cpu()), traj["f"].append(cmd.f.clone().cpu()), traj["alive"].append(alive.cpu())
            traj["act"].append(u.sim.data.act[:, cmd._act_adr].cpu())  # muscle activations, in the robot's actuator order
        a = policy(obs)
        if action_noise > 0:  # as the runner collects: noise on the excitations, clipped
            a = (a + action_noise * torch.randn(a.shape, generator=gen, device=a.device)).clamp(0.0, 1.0)
        obs, rew, dones, _ = env.step(a)
        ret += torch.where(alive, rew, 0.0)
        failed |= u.reset_terminated & alive
        done_eval |= dones.bool()
        reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:  # finished clips restart but no longer count
            obs = TensorDict(u.reset(env_ids=reset_ids)[0], batch_size=[C])
    norm = ret / ((m.clip_len - 1).float() * cmd.max_step_reward)
    res = {
        "norm_return": norm.mean().item(),
        "norm_return_framew": (ret.sum() / ((m.clip_len - 1).sum() * cmd.max_step_reward)).item(),
        "n_failed": int(failed.sum().item()),
        "failed": [m.names[i] for i in failed.nonzero().flatten().tolist()],
        "per_clip": {m.names[i]: round(norm[i].item(), 4) for i in range(C)},
    }
    if record:
        ref_q, _ = cmd.kin.full_joint_state(m.q, m.qd)
        ref_quat = mat_to_quat(m.root_R @ cmd.kin.R_off.T)
        np.savez_compressed(record, names=np.array(m.names), **{k: torch.stack(v).numpy() for k, v in traj.items()},
                            ref_qpos=torch.cat((m.root_p, ref_quat, ref_q), -1).cpu().numpy(),
                            clip_start=m.clip_start.cpu().numpy(), clip_len=m.clip_len.cpu().numpy(), fps=1.0 / u.step_dt)  # fmt: skip
    return res


@torch.no_grad()
def evaluate_many(env, runner, policies: list, action_noise: float = 0.0, seed: int = 0) -> list[dict]:
    """:func:`evaluate` of several policies at once, in an env built with ``copies >= len(policies)``: policy k drives
    worlds k C .. k C + C - 1, one per clip; worlds of unused copies follow the last policy and are ignored."""
    u, cmd = env.unwrapped, runner.cmd
    m, C = cmd.motion, len(cmd.motion.names)
    N = u.num_envs
    K = N // C
    assert N == K * C and 1 <= len(policies) <= K, (N, C, len(policies))
    pols = list(policies) + [policies[-1]] * (K - len(policies))
    obs = TensorDict(u.reset(env_ids=torch.arange(N, device=u.device))[0], batch_size=[N])
    gen = torch.Generator(device=u.device).manual_seed(seed)
    ret = torch.zeros(N, device=u.device)
    done_eval = torch.zeros(N, dtype=torch.bool, device=u.device)
    failed = torch.zeros_like(done_eval)
    while not bool(done_eval.all()):
        alive = ~done_eval
        a = torch.cat([p(obs["policy"][k * C:(k + 1) * C], obs["goal"][k * C:(k + 1) * C]) for k, p in enumerate(pols)])
        if action_noise > 0:
            a = (a + action_noise * torch.randn(a.shape, generator=gen, device=a.device)).clamp(0.0, 1.0)
        obs, rew, dones, _ = env.step(a)
        ret += torch.where(alive, rew, 0.0)
        failed |= u.reset_terminated & alive
        done_eval |= dones.bool()
        reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:  # finished clips restart but no longer count
            obs = TensorDict(u.reset(env_ids=reset_ids)[0], batch_size=[N])
    out = []
    for k in range(len(policies)):
        rk, fk = ret[k * C:(k + 1) * C], failed[k * C:(k + 1) * C]
        norm = rk / ((m.clip_len - 1).float() * cmd.max_step_reward)
        out.append({
            "norm_return": norm.mean().item(),
            "norm_return_framew": (rk.sum() / ((m.clip_len - 1).sum() * cmd.max_step_reward)).item(),
            "n_failed": int(fk.sum().item()),
            "failed": [m.names[i] for i in fk.nonzero().flatten().tolist()],
            "per_clip": {m.names[i]: round(norm[i].item(), 4) for i in range(C)},
        })  # fmt: skip
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="Mjlab-SuperMimic-Flat-MS-Human-700")
    ap.add_argument("--motion-file", required=True)
    ap.add_argument("--checkpoint", default=None, help="evaluate one checkpoint")
    ap.add_argument("--run-dir", default=None, help="evaluate every model_*.pt of a run, appending to eval.csv")
    ap.add_argument("--watch", action="store_true", help="keep polling --run-dir for new checkpoints")
    ap.add_argument("--record", default=None, help="save trajectories (npz) of --checkpoint")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--action-noise", type=float, default=0.0, help="exploration noise std on the excitations (training: 0.1)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the action noise")
    ap.add_argument("--batch", type=int, default=1, help="--run-dir: checkpoints evaluated at once (K x C worlds)")
    ap.add_argument("--skip-before-min", type=float, default=0.0,
                    help="--run-dir: skip checkpoints saved before this many minutes of training (not written to eval.csv)")
    args = ap.parse_args()
    assert args.checkpoint or args.run_dir, "pass --checkpoint or --run-dir"
    env, runner = build(args.task, args.motion_file, args.device, copies=1 if args.checkpoint else max(1, args.batch))
    C = len(runner.motion.names)
    if args.checkpoint:
        runner.load(args.checkpoint, load_cfg={"actor": True})
        res = evaluate(env, runner, args.record, args.action_noise, args.seed)
        print(json.dumps({k: v for k, v in res.items() if k != "per_clip"}, indent=1))
        return
    out_csv = os.path.join(args.run_dir, "eval.csv")
    done = set()
    if os.path.exists(out_csv):
        done = {int(line.split(",")[0]) for line in open(out_csv).read().strip().splitlines()[1:]}
    else:
        open(out_csv, "w").write("iteration,train_wall_h,saved_at,eval_at,norm_return,norm_return_framew,n_failed,failed\n")
    pending = []  # (iteration, path, checkpoint dict, policy): evaluated K at a time

    def flush():
        if not pending:
            return
        t0 = time.time()
        results = evaluate_many(env, runner, [p[3] for p in pending], action_noise=args.action_noise, seed=args.seed)
        for (it, c, ck, _), res in zip(pending, results, strict=True):
            line = (f"{it},{ck['wall_time'] / 3600:.4f},{ck['saved_at']},{time.strftime('%Y-%m-%d %H:%M:%S')},{res['norm_return']:.4f},"
                    f"{res['norm_return_framew']:.4f},{res['n_failed']},{'|'.join(res['failed'])}\n")  # fmt: skip
            open(out_csv, "a").write(line)
            json.dump(res, open(c.replace(".pt", "_eval.json"), "w"), indent=1)
            print(f"[eval] it {it} ({ck['wall_time'] / 3600:.2f} h): norm return {res['norm_return']:.4f}, "
                  f"failed {res['n_failed']}/{C} ({(time.time() - t0) / len(pending):.0f}s, {len(pending)} at once)", flush=True)  # fmt: skip
            done.add(it)
        pending.clear()

    while True:
        for c in sorted(glob.glob(os.path.join(args.run_dir, "model_*.pt"))):
            it = int(os.path.basename(c)[6:-3])
            if it in done or any(p[0] == it for p in pending):
                continue
            ck = runner.load(c, load_cfg={"actor": True})
            if ck["wall_time"] / 60 < args.skip_before_min:
                print(f"[eval] it {it} ({ck['wall_time'] / 60:.1f} min): skipped (before {args.skip_before_min:g} min)", flush=True)
                done.add(it)
                continue
            pending.append((it, c, ck, copy.deepcopy(runner.deployed).eval()))
            if len(pending) == max(1, args.batch):
                flush()
        flush()  # a partial batch: what is there now
        if not args.watch:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
