"""Reset bins contain transitions, while terminal frames remain valid scoring references."""

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from muscle_supermimic.tasks.tracking.mdp.commands import MotionCommand
from muscle_supermimic.tasks.tracking.mdp.heads import StepCtx, TrackingHeads
from muscle_supermimic.tasks.tracking.mdp.kinematics import SkeletonKinematics
from muscle_supermimic.tasks.tracking.mdp.motion import MotionLib


@pytest.fixture(scope="module")
def kin() -> SkeletonKinematics:
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><worldbody><body name="pelvis"><freejoint name="root"/>'
        '<geom size=".1"/><body name="link" pos="0 0 .2"><joint name="q" range="-1 1"/>'
        '<geom size=".05"/><body name="coupled"><joint name="qc" range="-1 1"/>'
        '<geom size=".05"/></body></body></body></worldbody>'
        '<equality><joint joint1="qc" joint2="q" polycoef="0 1 0 0 0"/></equality></mujoco>'
    )
    return SkeletonKinematics(("pelvis", "link"), "cpu", model)


def _clip(path: Path, frames: int, fps: float = 50.0) -> str:
    p = np.zeros((frames, 3))
    p[:, 0] = np.arange(frames) / fps
    np.savez(
        path,
        fps=fps,
        root_pos=p,
        root_quat=np.tile([1.0, 0, 0, 0], (frames, 1)),
        joint_pos=np.zeros((frames, 1)),
        joint_names=np.array(["q"]),
    )
    return str(path)


@pytest.mark.parametrize("frames", [2, 50, 51, 100, 101, 151])
@pytest.mark.parametrize("masked", [False, True])
def test_sampled_resets_always_have_a_same_clip_successor(
    tmp_path: Path,
    kin: SkeletonKinematics,
    frames: int,
    masked: bool,
) -> None:
    motion = MotionLib([_clip(tmp_path / "clip.npz", frames)], kin, 0.02, "cpu", bin_frames=50)
    cmd = object.__new__(MotionCommand)
    cmd._env = SimpleNamespace(device="cpu", num_envs=32)
    cmd.cfg = SimpleNamespace(sampling_mode="uniform")
    cmd.motion = motion
    probabilities = torch.zeros(motion.num_bins)
    probabilities[-1] = 1.0  # force the final bin, including the exact-second boundary
    cmd.reset_probs = lambda gate=None: probabilities
    torch.manual_seed(1)
    f, bins = cmd.sample_frames_masked(torch.tensor(True)) if masked else cmd.sample_frames(32)
    assert torch.equal(bins, torch.full_like(bins, motion.num_bins - 1))
    assert (f < motion.clip_end[motion.clip_id[f]]).all()
    torch.testing.assert_close(motion.clip_id[f + 1], motion.clip_id[f])
    assert torch.isfinite(motion.ref_state(f + 1)[0]).all()
    heads = TrackingHeads(kin, motion, None, lookahead=2, stride=1)
    assert torch.isfinite(heads.goal(StepCtx((), f + 1))).all()


def test_terminal_scoring_frame_keeps_the_preceding_reset_bin(tmp_path: Path, kin: SkeletonKinematics) -> None:
    motion = MotionLib([_clip(tmp_path / "clip.npz", 101)], kin, 0.02, "cpu", bin_frames=50)
    assert motion.bin_start.tolist() == [0, 50]
    assert int(motion.frame_bin[-1]) == int(motion.frame_bin[-2]) == 1
    assert int(motion.clip_end[0]) == 100
    torch.testing.assert_close(motion.root_p[-1], torch.tensor([2.0, 0, 0]))


def test_each_clip_keeps_its_own_terminal_bookkeeping_bin(tmp_path: Path, kin: SkeletonKinematics) -> None:
    files = [_clip(tmp_path / f"clip{i}.npz", n) for i, n in enumerate((101, 2, 51))]
    motion = MotionLib(files, kin, 0.02, "cpu", bin_frames=50)
    assert motion.bin_start.tolist() == [0, 50, 101, 103]
    for c, end in enumerate(motion.clip_end):
        assert int(motion.bin_clip[motion.frame_bin[end]]) == c
        assert motion.frame_bin[end] == motion.frame_bin[end - 1]


@pytest.mark.parametrize(("frames", "fps"), [(1, 50.0), (2, 100.0)])
def test_clip_without_a_control_transition_is_rejected(
    tmp_path: Path,
    kin: SkeletonKinematics,
    frames: int,
    fps: float,
) -> None:
    path = _clip(tmp_path / "short.npz", frames, fps)
    with pytest.raises(ValueError, match="at least two frames"):
        MotionLib([path], kin, 0.02, "cpu")
