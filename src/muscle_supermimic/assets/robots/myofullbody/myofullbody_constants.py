"""Static facts about MuscleMimic's MyoFullBody whole-body musculoskeletal model.

The model is not vendored: as in MuscleMimic, it comes from the ``musclemimic-models`` package
(Apache-2.0; its NOTICE lists the MyoSuite / OpenSim sources it derives from and their licenses) via
``musclemimic_models.get_xml_path("myofullbody")``. ``myofullbody.xml`` has 102 bodies, 123 joints (the
free root, 112 hinges, 10 slides), 416 Hill-type muscles on spatial tendons, and 51 ``<equality>``
joint constraints: lumbar rhythm, shoulder rhythm, knee/patella kinematics and two abdominal slides held
at zero.
"""

from __future__ import annotations

from pathlib import Path

import musclemimic_models

MYOFULLBODY_XML: Path = musclemimic_models.get_xml_path("myofullbody")

# The free joint sits on "Full Body", the frame that holds the pelvis (welded to it) and the spine; the
# root state is that frame's, which shares its origin with the pelvis.
ROOT_BODY: str = "Full Body"
FREE_JOINT: str = "root"

# Upstream defaults the root to the identity, which faces -y. The entity turns it +90 deg about z so that
# its default pose faces +x (the convention of the canonical root frame), standing with feet on the floor.
INIT_POS: tuple[float, float, float] = (0.0, 0.0, 1.0)
INIT_ROT: tuple[float, float, float, float] = (0.7071068, 0.0, 0.0, 0.7071068)

# Muscle tendons are `<muscle>_tendon` (right side and midline) or `<muscle>_tendon_left`; the other eight
# tendons are inert glenohumeral ligaments (no stiffness, damping or range).
MUSCLE_TENDON_EXPR: tuple[str, ...] = (r".*_tendon", r".*_tendon_left")

# The joints and muscles MuscleMimic removes with `disable_fingers` (its default), from MuscleMimic's
# musclemimic/environments/humanoids/myofullbody.py (github.com/amathislab/musclemimic, Apache-2.0).
# The fingers end up welded to the hands in the neutral pose; the wrists stay.
FINGER_JOINTS: tuple[str, ...] = tuple(
    f"{name}_{side}"
    for side in ("r", "l")
    for name in (
        "cmc_flexion", "cmc_abduction", "mp_flexion", "ip_flexion",
        "mcp2_flexion", "mcp2_abduction", "mcp3_flexion", "mcp3_abduction",
        "mcp4_flexion", "mcp4_abduction", "mcp5_flexion", "mcp5_abduction",
        "md2_flexion", "md3_flexion", "md4_flexion", "md5_flexion",
        "pm2_flexion", "pm3_flexion", "pm4_flexion", "pm5_flexion",
    )
)  # fmt: skip
_RIGHT_FINGER_MUSCLES: tuple[str, ...] = (
    "FDS2", "FDS3", "FDS4", "FDS5", "FDP2", "FDP3", "FDP4", "FDP5",
    "EDC2", "EDC3", "EDC4", "EDC5", "EDM", "EIP", "EPL", "EPB", "FPL", "APL", "OP",
    "RI2", "RI3", "RI4", "RI5", "LU_RB2", "LU_RB3", "LU_RB4", "LU_RB5", "UI_UB2", "UI_UB3", "UI_UB4", "UI_UB5",
)  # fmt: skip
FINGER_MUSCLES: tuple[str, ...] = _RIGHT_FINGER_MUSCLES + tuple(f"{n}_left" for n in _RIGHT_FINGER_MUSCLES)
