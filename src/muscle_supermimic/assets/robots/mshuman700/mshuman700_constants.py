"""Static facts about the MS-Human-700 whole-body musculoskeletal model.

The model is not vendored: it lives in the ``third_party/MS-Human-700`` git submodule
(``MS_HUMAN_700_ROOT`` overrides the location). ``MS-Human-700.xml`` is the primary whole-body
model: 81 bodies, 85 joints (6 of them the OpenSim-style pelvis DoF), 700 Hill-type muscles on
wrapping spatial tendons, and 42 ``<equality>`` joint couplers (knee/patella kinematics, shoulder
rhythm, wrist), which leave 37 independent joint DoF besides the root.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT_ENV_VAR: str = "MS_HUMAN_700_ROOT"
SUBMODULE_RELPATH: Path = Path("third_party") / "MS-Human-700"


def _resolve_root() -> Path:
    override = os.environ.get(ROOT_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / SUBMODULE_RELPATH).is_dir():
            return parent / SUBMODULE_RELPATH
    # src/muscle_supermimic/assets/robots/mshuman700/<this file> -> repo root
    return here.parents[5] / SUBMODULE_RELPATH


MSHUMAN700_ROOT: Path = _resolve_root()
MSHUMAN700_XML: Path = MSHUMAN700_ROOT / "MS-Human-700.xml"


def require_mshuman700_xml(path: Path = MSHUMAN700_XML) -> Path:
    """Path to the model file, or an error telling the user how to get it (not asserted at import time,
    so task registration on ``import mjlab`` survives an uninitialised submodule)."""
    if path.is_file():
        return path
    raise FileNotFoundError(
        f"MS-Human-700 model not found at {path}.\n"
        "It ships as a git submodule. Fetch it with:\n"
        "    git submodule update --init --recursive\n"
        f"or point {ROOT_ENV_VAR} at an existing MS-Human-700 checkout."
    )


# The pelvis carries the root. Upstream models it as three slides + three Euler hinges (OpenSim
# convention); the entity replaces them with one free joint.
ROOT_BODY: str = "pelvis"
PELVIS_JOINTS: tuple[str, ...] = ("pelvis_tz", "pelvis_ty", "pelvis_tx", "pelvis_tilt", "pelvis_list", "pelvis_rotation")
FREE_JOINT: str = "root"

# Every muscle drives exactly one spatial tendon named `<muscle>_tendon`; there are no other tendons.
MUSCLE_TENDON_EXPR: tuple[str, ...] = (r".*_tendon",)

# Upstream standing pose: pelvis 0.95 m above the floor, its OpenSim frame (x forward, y up, z right)
# rotated +90 deg about world x so that y points up.
INIT_POS: tuple[float, float, float] = (0.0, 0.0, 0.95)
INIT_ROT: tuple[float, float, float, float] = (0.7071068, 0.7071068, 0.0, 0.0)

# Root frame convention used everywhere downstream: R_canonical = R_pelvis @ ROOT_OFFSET, i.e. x
# forward, y left, z up, the identity when standing upright facing +x (as for the G1's pelvis).
