"""Wall-clock of one training iteration, split into collection, world-model updates and the policy update.

Builds the task's env and the SuperMimic runner with the registered settings plus ``--set`` (runner cfg) and
``--env-set`` (motion command cfg) and ``--sim-set`` (MuJoCo solver cfg, e.g. ``iterations=50 ls_iterations=25``) overrides, trains ``--warmup`` iterations untimed (compilation, buffer fill;
policy updates from the first iteration) and then times ``--iters`` iterations piece by piece, the device
synchronized around each piece. Reports seconds per iteration and torch's peak memory.

  uv run python scripts/bench_iteration.py --task Mjlab-SuperMimic-Flat-MyoFullBody --motion-file data/myofullbody/walk.npz
  uv run python scripts/bench_iteration.py ... --set pi_batch=2048 wm_batches=5 wm_hidden=1024,1024,1024 pi_opt=muon \\
      wm_opt=muon fused_adam=true no_sync=true pi_ema=0.995 empty_cache=false cuda_graph=true --env-set bin_fix=true fail_reset=0.3

``--set cuda_graph=true`` times the updates as CUDA graphs (captured during the warm-up, as in training);
``--set collect_graph=true`` the collection steps through the graphed simulator wrapper (collector.py);
``--set overlap_collect=true`` (with both) times the runner's own ``iterate``, the collection on a second stream
concurrently with the updates, as one piece.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict

import torch


def parse_override(default, v: str):
    """A ``KEY=VALUE`` value in the type of the cfg field's default (tuples comma-separated)."""
    if isinstance(default, bool):
        return v.lower() in ("1", "true", "yes")
    if isinstance(default, tuple):
        return tuple(type(default[0])(x) for x in v.split(","))
    return type(default)(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="Mjlab-SuperMimic-Flat-MyoFullBody")
    ap.add_argument("--motion-file", required=True)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--warmup", type=int, default=10, help="untimed iterations first (compilation, buffer fill)")
    ap.add_argument("--iters", type=int, default=20, help="timed iterations")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="runner cfg overrides, e.g. pi_batch=2048")
    ap.add_argument("--env-set", nargs="*", default=[], metavar="KEY=VALUE", help="motion command cfg overrides, e.g. bin_fix=true")
    ap.add_argument("--sim-set", nargs="*", default=[], metavar="KEY=VALUE", help="MuJoCo solver cfg overrides, e.g. iterations=50")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--profile", type=int, default=0,
                    help="after the timing, profile this many more iterations (best without CUDA graphs) and print the top CUDA kernels")
    args = ap.parse_args()

    import mjlab.tasks  # noqa: F401  (registers the tasks, ours included via the entry point)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


    torch.manual_seed(0)
    cfg = load_env_cfg(args.task)
    cmd_cfg = cfg.commands["motion"]
    cmd_cfg.motion_file = args.motion_file
    for kv in args.env_set:
        k, v = kv.split("=", 1)
        setattr(cmd_cfg, k, parse_override(getattr(cmd_cfg, k), v))
    for kv in args.sim_set:
        k, v = kv.split("=", 1)
        setattr(cfg.sim.mujoco, k, parse_override(getattr(cfg.sim.mujoco, k), v))
    cfg.scene.num_envs = N = args.num_envs
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=args.device))
    rl_default = load_rl_cfg(args.task)
    overrides = {k: parse_override(getattr(rl_default, k), v) for k, v in (kv.split("=", 1) for kv in args.set)}
    rl = asdict(rl_default)
    rl.update(wm_only_iters=0, **overrides)
    runner = load_runner_cls(args.task)(env, rl, device=args.device)
    c = runner.cfg
    cuda = str(args.device).startswith("cuda")
    sync = torch.cuda.synchronize if cuda else (lambda: None)

    # learn() without its logging: the pieces of an iteration, timed one by one
    runner.init_collection()  # with collect_graph also the collector (captured during the warm-up, as in training)
    from muscle_supermimic.rl.supermimic.collector import STATS

    stats = {k: [] for k in STATS}
    runner.collect(max(c.warmup_steps, c.wm_window + runner.hist + 1), stats)
    from muscle_supermimic.rl.supermimic.runner import set_lr

    for o in (runner.opt_wm, runner.opt_pi, runner.opt_v):
        set_lr(o, c.lr)
    wm_update, policy_update = runner.wm_update, runner.policy_update
    if c.cuda_graph:  # as learn(): the updates captured after three warm-up calls each (the policy's after the EMA copy)
        if runner.buffer.ptr_t is None:
            runner.buffer.use_device_ptr()
        wm_update, policy_update = runner.make_graphs()
    step_acc, t = 0.0, {"collect": 0.0, "wm": 0.0, "policy": 0.0, "overlapped": 0.0}
    n_overlapped = 0
    t0 = time.time()
    for it in range(args.warmup + args.iters):
        timed = it >= args.warmup
        if it == args.warmup:
            sync()
            if cuda:
                print(f"warm-up peak: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB allocated, "
                      f"{torch.cuda.max_memory_reserved() / 2**30:.1f} GB reserved by torch", flush=True)  # fmt: skip
                torch.cuda.reset_peak_memory_stats()
            print(f"warm-up done in {time.time() - t0:.0f} s; timing {args.iters} iterations", flush=True)
            t0 = time.time()
        step_acc += 1.0 / c.utd
        n = int(step_acc)
        step_acc -= n
        if c.overlap_collect or c.overlap_wm:  # the runner's iteration, as learn() runs it
            agg = {k: [] for k in ("wm", "wm8", "wmt", "pi", "v", "ir", "gn", "surv", "clip")}
            sync()
            t1 = time.time()
            overlapped = runner.can_overlap(it) or runner.can_overlap_wm(it)
            runner.iterate(it, n, stats, agg, wm_update, policy_update)
            sync()
            if timed:
                t["overlapped"] += time.time() - t1
                n_overlapped += overlapped
            stats = {k: [] for k in stats}
            continue
        sync()
        t1 = time.time()
        runner.collect(n, stats)
        sync()
        t2 = time.time()
        for _ in range(c.wm_batches):
            wm_update()
        sync()
        t3 = time.time()
        policy_update()
        if c.empty_cache and cuda:
            torch.cuda.empty_cache()
        sync()
        t4 = time.time()
        if timed:
            t["collect"] += t2 - t1
            t["wm"] += t3 - t2
            t["policy"] += t4 - t3
        stats = {k: [] for k in stats}
    total = time.time() - t0
    mem = torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0
    sim_steps = sum(int(k) for k in [args.iters / c.utd])
    print(f"\n{args.task}, {N} envs, {args.iters} timed iterations, overrides {overrides or 'none'}, "
          f"env overrides {args.env_set or 'none'}, solver {cfg.sim.mujoco.integrator} {cfg.sim.mujoco.iterations}/"
          f"{cfg.sim.mujoco.ls_iterations}, device {args.device}")  # fmt: skip
    qpos = runner.uenv.sim.data.qpos._tensor
    print(f"simulator state finite in {int(torch.isfinite(qpos).all(-1).sum())} of {N} envs")
    print(f"{'piece':10s} {'s / iteration':>14s} {'share':>7s}")
    for k, v in t.items():
        if v > 0:
            print(f"{k:10s} {v / args.iters:14.3f} {v / total:7.1%}")
    if c.overlap_collect or c.overlap_wm:
        print(f"{n_overlapped} of {args.iters} timed iterations ran overlapped (collection and/or world-model updates)")
    if args.profile > 0 and cuda:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(args.profile):
                runner.collect(int(round(1.0 / c.utd)), stats)
                for _ in range(c.wm_batches):
                    wm_update()
                policy_update()
            sync()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30, max_name_column_width=70))
    print(f"{'total':10s} {total / args.iters:14.3f} {'':7s}   ({args.iters / total:.3f} it/s; collection = {sim_steps / args.iters:.1f} "
          f"sim steps of {N} envs per iteration; torch peak {mem:.1f} GB)")  # fmt: skip


if __name__ == "__main__":
    main()
