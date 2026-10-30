"""Export a checkpoint for the browser demo (docs/website/demo/), and record the reference closed loop it is checked against.

The demo steps the task's own scene model with MuJoCo's WebAssembly build and runs the policy's decoder with
onnxruntime-web. The encoder sees only the reference, so its latent is evaluated here for every frame of the
clips. The observation is recomputed in JavaScript (docs/website/demo/tracker.mjs) from MuJoCo quantities; this
script writes what that needs, the large files gzipped for the page to download:

  model.mjb.gz    the task's scene model with its solver settings, as the simulator runs it
  policy.onnx.gz  the decoder: (z, obs) -> excitations, with the observation normalization and the sigmoid; its
                  weight matrices are stored in float16 (and cast back on load) unless ``--fp32``
  clips/<k>.bin.gz  per frame of clip k, float32: the latent z, then the reference qpos (root + every joint); one
                  file per clip, so the page downloads a clip when it is chosen
  init.bin        per clip, float64: the qpos and qvel a reset writes at the clip's first frame
  meta.json       addresses, coupled-joint polynomials, tracked bodies, observation layout, the clip table (with
                  each clip's file) and the file names above

With ``--reference``, every clip also plays from its first frame under native MuJoCo (the C library the
WebAssembly build compiles), with the task's own observation code and the torch policy, as ``ms-eval`` plays
it on MuJoCo-Warp. Per step it records the observation and the excitation (reference.bin, float32) and the
state they were computed on (reference_state.bin, float64: qpos, qvel, act and the solver's warm start), so
web/test/align.mjs can check the JavaScript observation and policy on the same states, and its closed loop. The
reference runs the policy with the weights the export stores (float16-rounded unless ``--fp32``).
``--check-env N`` first replays N steps of the mjlab env and checks that the native observation of each
of its states equals the env's.

  uv run ms-export-web --task Mjlab-SuperMimic-Flat-MyoFullBody --checkpoint <model.pt> \
      --motion-file data/musclemimic/demo_clips.txt --out docs/website/demo/data/myofullbody
  MS_HUMAN_700_ROOT=<MS-Human-700> uv run ms-export-web --task Mjlab-SuperMimic-Flat-MS-Human-700-Locked-Hands \
      --checkpoint <model.pt> --motion-file data/mshuman700_kit/demo_clips.txt --out docs/website/demo/data/mshuman700
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil

import mujoco
import numpy as np
import torch
import torch.nn as nn

from muscle_supermimic.utils.rotations import mat_to_quat, quat_to_mat

from ..mdp.heads import StepCtx
from .evaluate import build


class Decoder(nn.Module):
    """The policy as the demo runs it: the encoder's latent is an input (precomputed per frame)."""

    def __init__(self, pi):
        super().__init__()
        self.dec, self.norm = pi.dec, pi.obs_norm

    def forward(self, z: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.dec(torch.cat((z, self.norm(obs)), -1)))


class NativeSim:
    """The env's step and observation on a native MuJoCo model (one clip at a time)."""

    def __init__(self, env, runner):
        self.u, self.cmd, self.runner = env.unwrapped, runner.cmd, runner
        self.m = self.u.sim.mj_model
        self.d = mujoco.MjData(self.m)
        cmd = self.cmd
        self.free_q, self.free_v = cmd._free_q.cpu().numpy(), cmd._free_v.cpu().numpy()
        self.q_adr, self.v_adr = cmd._q_adr.cpu().numpy(), cmd._v_adr.cpu().numpy()
        self.act_adr = cmd._act_adr.cpu().numpy()
        self.ctrl_ids = self.u.scene["robot"].indexing.ctrl_ids.cpu().numpy()
        self.touch_adr = np.array(
            [self.m.sensor(f"{cmd.cfg.touch_sensor_name}_{b}_force").adr[0] + 2 for b in cmd.touch_sensor.primary_names]
        )
        self.R_off = cmd.kin.R_off.cpu()
        self.decimation = self.u.cfg.decimation
        self.obs_terms = list(self.u.observation_manager.active_terms["policy"])
        self.rewards = [(t.func.analytic, t.weight * self.u.step_dt, {k: v for k, v in t.params.items() if k != "command_name"})
                        for t in self.u.cfg.rewards.values()]  # fmt: skip
        self.fails = [
            (t.func.analytic, t.params["threshold"]) for k, t in self.u.cfg.terminations.items() if k.startswith(("body", "root"))
        ]

    def state(self):
        """MotionCommand.read_state on the native data."""
        d, t = self.d, lambda x: torch.tensor(np.asarray(x), dtype=torch.float32)[None]
        Rb = quat_to_mat(t(d.qpos[self.free_q[3:7]]))
        w = (Rb @ t(d.qvel[self.free_v[3:6]])[..., None])[..., 0]
        touch = t(np.abs(d.sensordata[self.touch_adr]))
        return (t(d.qpos[self.free_q[:3]]), Rb @ self.R_off, t(d.qvel[self.free_v[:3]]), w, t(d.qpos[self.q_adr]),
                t(d.qvel[self.v_adr]), t(d.act[self.act_adr]), touch)  # fmt: skip

    def obs(self, f: int, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = StepCtx(self.state(), torch.tensor([f]), a)
        h = self.cmd.heads
        return torch.cat([getattr(h, n)(ctx) for n in self.obs_terms], -1), h.goal(ctx)

    def step(self, e: np.ndarray, f: int) -> tuple[float, bool]:
        """One control step under excitation e; reward and failure of the post-step state against frame f + 1."""
        self.d.ctrl[self.ctrl_ids] = np.clip(e, 0.0, 1.0)
        for _ in range(self.decimation):
            mujoco.mj_step(self.m, self.d)
        ctx = StepCtx(self.state(), torch.tensor([f + 1]))
        r = sum(fn(self.cmd.heads, ctx, **p).item() * w for fn, w, p in self.rewards)
        failed = any(bool(fn(self.cmd.heads, ctx, threshold=th)) for fn, th in self.fails)
        mujoco.mj_forward(self.m, self.d)  # the env's forward before its observation (touch sensors)
        return r, failed


FP16_MIN_SIZE = 4096  # tensors stored in float16: the decoder's weight matrices (its biases and the normalizer stay float32)


def round_weights_fp16(pi) -> None:
    """Round the decoder's large tensors to float16 in place, so the reference runs the weights the export stores."""
    with torch.no_grad():
        for p in pi.dec.parameters():
            if p.numel() >= FP16_MIN_SIZE:
                p.copy_(p.half().float())


def store_fp16(path: str) -> None:
    """Store the ONNX model's large float initializers as float16, each cast back to float32 in the graph."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    mp = onnx.load(path)
    casts = []
    for init in mp.graph.initializer:
        if init.data_type == TensorProto.FLOAT and int(np.prod(init.dims)) >= FP16_MIN_SIZE:
            name = init.name
            init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), name + "_fp16"))
            casts.append(helper.make_node("Cast", [name + "_fp16"], [name], to=TensorProto.FLOAT))
    for node in reversed(casts):
        mp.graph.node.insert(0, node)
    onnx.checker.check_model(mp)
    onnx.save(mp, path)


def gzip_file(path: str) -> str:
    """Gzip a file in place, without a timestamp: the same content re-exported gives the same bytes."""
    with open(path, "rb") as src, open(path + ".gz", "wb") as raw, gzip.GzipFile("", "wb", 9, raw, mtime=0) as dst:
        shutil.copyfileobj(src, dst)
    os.remove(path)
    return os.path.basename(path) + ".gz"


def export(env, runner, out: str, fp16: bool = True) -> dict:
    u, cmd, pi = env.unwrapped, runner.cmd, runner.pi.eval()
    mo, kin, m = cmd.motion, cmd.kin, u.sim.mj_model
    os.makedirs(out, exist_ok=True)
    mujoco.mj_saveModel(m, os.path.join(out, "model.mjb"), None)
    nz, nobs = pi.dec[0].in_features - pi.obs_norm.mean.numel(), pi.obs_norm.mean.numel()
    if fp16:
        round_weights_fp16(pi)
    torch.onnx.export(Decoder(pi).cpu(), (torch.zeros(1, nz), torch.zeros(1, nobs)), os.path.join(out, "policy.onnx"),
                      input_names=["z", "obs"], output_names=["excitation"], dynamo=False)  # fmt: skip
    if fp16:
        store_fp16(os.path.join(out, "policy.onnx"))

    # per frame: the latent of the reference goal and the reference qpos (as ms-eval records it)
    with torch.no_grad():
        f = torch.arange(len(mo.clip_id))
        z = torch.tanh(pi.enc(pi.goal_norm(cmd.heads.goal(StepCtx((), f)))))
    ref_q, _ = kin.full_joint_state(mo.q, mo.qd)
    ref_qpos = torch.cat((mo.root_p, mat_to_quat(mo.root_R @ kin.R_off.T), ref_q), -1)
    frames = np.concatenate((z.numpy(), ref_qpos.numpy()), 1).astype("<f4")
    shutil.rmtree(os.path.join(out, "clips"), ignore_errors=True)
    os.makedirs(os.path.join(out, "clips"))
    clips = []
    for k, (name, s, n) in enumerate(zip(mo.names, mo.clip_start.tolist(), mo.clip_len.tolist())):
        frames[s : s + n].tofile(os.path.join(out, "clips", f"{k}.bin"))
        f = "clips/" + gzip_file(os.path.join(out, "clips", f"{k}.bin"))
        clips.append({"name": name, "start": int(s), "length": int(n), "file": f, "size": os.path.getsize(os.path.join(out, f))})

    # per clip: the reset state the env writes at the clip's first frame (one env per clip)
    C = len(mo.names)
    u.reset(env_ids=torch.arange(C))
    qpos = u.sim.data.qpos.cpu().numpy().astype(np.float64)
    qpos[:, cmd._free_q[:3].cpu().numpy()] -= u.scene.env_origins.cpu().numpy()
    np.concatenate((qpos, u.sim.data.qvel.cpu().numpy()), 1).astype("<f8").tofile(os.path.join(out, "init.bin"))

    robot = u.scene["robot"].indexing
    jq = robot.joint_q_adr.long().cpu().numpy()  # every non-root joint, in entity (= kin.full_index) order
    meta = {
        "nq": int(m.nq), "nv": int(m.nv), "nu": int(m.nu), "na": int(m.na), "decimation": u.cfg.decimation, "timestep": float(m.opt.timestep),
        "free_q": cmd._free_q.tolist(), "free_v": cmd._free_v.tolist(),
        "q_adr": cmd._q_adr.tolist(), "v_adr": cmd._v_adr.tolist(), "act_adr": cmd._act_adr.tolist(),
        "ctrl_ids": robot.ctrl_ids.tolist(), "init_activation": cmd.cfg.init_activation,
        # every non-root joint's qpos / qvel address, filled from cat(independent, coupled)[full_index]
        "joint_q_adr": jq.tolist(), "joint_v_adr": robot.joint_v_adr.long().tolist(), "full_index": kin.full_index.tolist(),
        "coupled": {"src": kin.c_src.tolist(), "coef": kin.c_coef.tolist(), "q0": kin.c_q0.tolist(), "src_q0": kin.c_src_q0.tolist()},
        "R_off": kin.R_off.flatten().tolist(),
        "tracked_body_ids": [int(m.body(f"robot/{n}").id) for n in cmd.cfg.body_names],
        "touch_adr": [int(m.sensor(f"{cmd.cfg.touch_sensor_name}_{b}_force").adr[0]) + 2 for b in cmd.touch_sensor.primary_names],
        "obs_terms": [[n, d[0]] for n, d in zip(u.observation_manager.active_terms["policy"], u.observation_manager.group_obs_term_dim["policy"])],
        "nz": nz, "nobs": nobs, "frame_width": nz + ref_qpos.shape[1],
        "nmuscles": int(len(robot.ctrl_ids)),
        "clips": clips,  # start: the clip's first frame in the task's motion (the reference's frame index)
        "files": {"model": gzip_file(os.path.join(out, "model.mjb")), "policy": gzip_file(os.path.join(out, "policy.onnx")),
                  "init": "init.bin"},
    }  # fmt: skip
    meta["sizes"] = {k: os.path.getsize(os.path.join(out, f)) for k, f in meta["files"].items()}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    return meta


@torch.no_grad()
def check_env(env, runner, sim: NativeSim, steps: int) -> None:
    """The native observation of each state the env visits (written into the native data) equals the env's."""
    u, cmd = env.unwrapped, runner.cmd
    C = len(cmd.motion.names)
    obs = u.reset(env_ids=torch.arange(C))[0]
    policy = runner.get_inference_policy()
    dims = u.observation_manager.group_obs_term_dim["policy"]
    worst = np.zeros(len(dims))
    for _ in range(steps):
        a = policy(obs)
        obs = env.step(a)[0]
        assert np.allclose(u.sim.data.ctrl[:, sim.ctrl_ids].cpu().numpy(), a.clamp(0, 1).cpu().numpy())
        for c in range(C):
            d = sim.d
            d.qpos[:] = u.sim.data.qpos[c].cpu().numpy()
            d.qpos[sim.free_q[:3]] -= u.scene.env_origins[c].cpu().numpy()
            d.qvel[:], d.act[:] = u.sim.data.qvel[c].cpu().numpy(), u.sim.data.act[c].cpu().numpy()
            mujoco.mj_forward(sim.m, d)
            o, g = sim.obs(int(cmd.f[c]), u.action_manager.action[c : c + 1])
            err = (o[0] - obs["policy"][c]).abs()
            assert (g[0] - obs["goal"][c]).abs().max() < 1e-5
            worst = np.maximum(worst, [e.max().item() for e in err.split([x[0] for x in dims], -1)])
    names = u.observation_manager.active_terms["policy"]
    print("[check-env] max |native - env| per observation term over", steps, "steps x", C, "clips:")
    print("  " + ", ".join(f"{n} {w:.1e}" for n, w in zip(names, worst)))


@torch.no_grad()
def reference(env, runner, sim: NativeSim, meta: dict, out: str) -> None:
    pi, rec, states, res = runner.pi.eval(), [], [], []
    init = np.fromfile(os.path.join(out, "init.bin"), "<f8").reshape(len(meta["clips"]), -1)
    max_r = runner.cmd.max_step_reward
    for c, clip in enumerate(meta["clips"]):
        d = sim.d
        mujoco.mj_resetData(sim.m, d)
        d.qpos[:], d.qvel[:] = init[c, : meta["nq"]], init[c, meta["nq"] :]
        d.act[sim.act_adr] = meta["init_activation"]
        mujoco.mj_forward(sim.m, d)
        a, ret, failed, T = torch.zeros(1, meta["nu"]), 0.0, False, clip["length"] - 1
        for t in range(T):
            f = clip["start"] + t
            o, g = sim.obs(f, a)
            e = pi(o, g)
            rec.append(np.concatenate((o[0].numpy(), e[0].numpy())).astype("<f4"))
            states.append(np.concatenate((d.qpos, d.qvel, d.act, d.qacc_warmstart)))
            r, fail = sim.step(e[0].numpy(), f)
            ret, a = ret + r, e
            if fail:
                failed, T = True, t + 1
                break
        res.append({"name": clip["name"], "steps": T, "norm_return": ret / ((clip["length"] - 1) * max_r), "failed": failed})
        print(f"[reference] {clip['name']}: {T} steps, norm return {res[-1]['norm_return']:.4f}{', failed' if failed else ''}")
    np.stack(rec).tofile(os.path.join(out, "reference.bin"))
    np.stack(states).astype("<f8").tofile(os.path.join(out, "reference_state.bin"))
    meta["reference"] = {"rows": len(rec), "layout": ["obs", meta["nobs"], "excitation", meta["nu"]],
                         "state_layout": ["qpos", meta["nq"], "qvel", meta["nv"], "act", meta["na"], "qacc_warmstart", meta["nv"]],
                         "clips": res}  # fmt: skip
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    n = np.array([r["norm_return"] for r in res])
    print(f"[reference] {len(res)} clips: norm return {n.mean():.4f}, failed {sum(r['failed'] for r in res)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="Mjlab-SuperMimic-Flat-MyoFullBody")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--motion-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", action="store_true", help="record the native closed loop of every clip")
    ap.add_argument("--check-env", type=int, default=0, help="compare native and env observations over N env steps")
    ap.add_argument("--fp32", action="store_true", help="keep the decoder's weights in float32")
    args = ap.parse_args()
    env, runner = build(args.task, args.motion_file, "cpu")
    runner.load(args.checkpoint, load_cfg={"actor": True})
    meta = export(env, runner, args.out, fp16=not args.fp32)
    sim = NativeSim(env, runner)
    if args.check_env:
        check_env(env, runner, sim, args.check_env)
    if args.reference:
        reference(env, runner, sim, meta, args.out)


if __name__ == "__main__":
    main()
