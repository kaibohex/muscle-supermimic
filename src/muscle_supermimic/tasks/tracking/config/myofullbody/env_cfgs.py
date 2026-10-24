"""MyoFullBody flat-ground tracking environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from muscle_supermimic.assets.robots import get_myofullbody_cfg
from muscle_supermimic.tasks.tracking.mdp import MotionCommandCfg
from muscle_supermimic.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

# The bodies of MuscleMimic's 17 mimic sites (each at its body's origin), pelvis first: pelvis, upper
# lumbar spine, head, shoulders, elbows, wrists, hips, knees, ankles, toes.
TRACKED_BODIES: tuple[str, ...] = (
    "pelvis", "lumbar1", "head",
    "humerus_l", "ulna_l", "lunate_l",
    "humerus_r", "ulna_r", "lunate_r",
    "femur_l", "tibia_l", "talus_l", "toes_l",
    "femur_r", "tibia_r", "talus_r", "toes_r",
)  # fmt: skip
# The bodies of MyoFullBody's touch sensor sites, in MuscleMimic's order: r_foot, r_toes, l_foot, l_toes.
FOOT_BODIES: tuple[str, ...] = ("calcn_r", "toes_r", "calcn_l", "toes_l")


def myofullbody_flat_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """MyoFullBody as MuscleMimic trains it: fingers disabled (31 joint DoF, 354 muscles), self-contact pairs kept.

    Args:
      play: auto-reset, one env, every clip from its first frame (mjlab's ``play``).
    """
    cfg = make_tracking_env_cfg()
    cfg.scene.entities = {"robot": get_myofullbody_cfg(self_collisions=True, disable_fingers=True)}
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.body_names = TRACKED_BODIES
    cfg.viewer.body_name = "pelvis"
    # foot touch as the ground's net force on the bodies of the touch sensor sites (what those sensors read on
    # flat ground), the same reading as MS-Human-700's
    cfg.scene.sensors = (
        ContactSensorCfg(
            name=motion_cmd.touch_sensor_name,
            primary=ContactMatch(mode="body", pattern=FOOT_BODIES, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("force",),
            reduce="netforce",
        ),
    )
    # MuscleMimic's per-env MuJoCo-Warp budget for this model (self-contact pairs + 51 equality rows)
    cfg.sim.nconmax, cfg.sim.njmax = 96, 768
    cfg.scene.num_envs = 4096  # supermusclemimic (and paper Table A.5)
    if play:
        # mjlab's play loop expects auto-reset; play every clip from its first frame
        cfg.scene.num_envs = 1
        cfg.auto_reset = True
        cfg.episode_length_s = int(1e9)
        motion_cmd.sampling_mode = "start"
        cfg.terminations.pop("time_out")
    return cfg
