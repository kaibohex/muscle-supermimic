"""Which clips a motion file names: a clip, a directory, or a list; and MuscleMimic's KIT split."""

import os

import pytest

from muscle_supermimic.tasks.tracking.mdp.motion import motion_files

SPLIT = os.path.join(os.path.dirname(__file__), "..", "data", "musclemimic")


def test_motion_files(tmp_path):
    for rel in ("a/1.npz", "a/b/2.npz", "3.npz"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).touch()
    root = str(tmp_path)
    assert motion_files(root) == [os.path.join(root, r) for r in ("3.npz", "a/1.npz", "a/b/2.npz")]
    assert motion_files(os.path.join(root, "3.npz")) == [os.path.join(root, "3.npz")]
    # a list keeps its order, resolves against its own directory and skips comments and blank lines
    (tmp_path / "a" / "clips.txt").write_text("# two clips\nb/2.npz\n\n1.npz  # the first\n")
    assert motion_files(str(tmp_path / "a" / "clips.txt")) == [os.path.join(root, "a", r) for r in ("b/2.npz", "1.npz")]
    (tmp_path / "bad.txt").write_text("3.npz\nnope.npz\n")
    with pytest.raises(FileNotFoundError, match="1 of 2 listed clips missing"):
        motion_files(str(tmp_path / "bad.txt"))


def _entries(name):
    with open(os.path.join(SPLIT, name)) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def test_musclemimic_kit_split():
    train, test = _entries("kit_kinesis_train.txt"), _entries("kit_kinesis_test.txt")
    assert (len(train), len(set(train)), len(test), len(set(test))) == (972, 972, 108, 108)
    assert not set(train) & set(test)
    assert all(e.startswith("MyoFullBody/gmr/KIT/") and e.endswith(".npz") for e in train + test)
