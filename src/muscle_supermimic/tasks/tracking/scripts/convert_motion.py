"""Convert a clip to the canonical motion format, optionally tiling a cyclic clip.

Input: an MS-Human-700 native clip (``qpos_traj`` (T,85) + ``framerate``, e.g. msgym's
``motion_data/walking_gait.npz``) or an already canonical one. A cyclic clip (one gait cycle whose
last frame precedes the first) can be repeated ``--cycles`` times, advancing the root by one stride per
cycle (the clip's net displacement extrapolated by one frame; msgym's ``stride`` key disagrees with
its own frames, so it is not used).

  uv run ms-convert-motion walking_gait.npz data/motions/walk.npz --cycles 8
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from muscle_supermimic.assets.robots.mshuman700 import ModelInfo, compiled_model, root_offset
from muscle_supermimic.tasks.tracking.mdp.motion import load_clip


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--cycles", type=int, default=1, help="repeat a cyclic clip this many times")
    args = ap.parse_args()

    m = compiled_model()
    info = ModelInfo(m)
    fps, p, quat, q = load_clip(args.input, info, root_offset(m))
    if args.cycles > 1:
        T = len(p)
        d = (p[-1] - p[0]) * T / (T - 1)
        d[2] = 0.0
        p = np.concatenate([p + k * d for k in range(args.cycles)])
        quat, q = np.tile(quat, (args.cycles, 1)), np.tile(q, (args.cycles, 1))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(args.output, fps=fps, root_pos=p, root_quat=quat, joint_pos=q, joint_names=np.array(info.indep_names))
    print(f"{args.output}: {len(p)} frames at {fps:g} fps ({len(p) / fps:.2f} s), root z {p[:, 2].min():.3f}..{p[:, 2].max():.3f} m")


if __name__ == "__main__":
    main()
