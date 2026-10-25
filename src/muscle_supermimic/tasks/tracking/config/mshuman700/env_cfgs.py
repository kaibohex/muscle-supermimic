"""MS-Human-700 flat-ground tracking environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from muscle_supermimic.assets.robots import get_mshuman700_cfg
from muscle_supermimic.tasks.tracking.mdp import MotionCommandCfg
from muscle_supermimic.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

# MuscleMimic's 17 mimic sites (pelvis first: the main site) as body frames, whose origins sit at the joints the sites
# mark: hips, knees, ankles, toes, the trunk above the lumbar/thoracic joints, head, shoulders, elbows, wrists.
TRACKED_BODIES: tuple[str, ...] = (
    "pelvis",
    "femur_r", "tibia_r", "calcn_r", "toes_r",
    "femur_l", "tibia_l", "calcn_l", "toes_l",
    "thoracic12", "head_neck",
    "humerus_r", "ulna_r", "hand_r",
    "humerus_l", "ulna_l", "hand_l",
)  # fmt: skip
FOOT_BODIES: tuple[str, ...] = ("calcn_r", "toes_r", "calcn_l", "toes_l")


def mshuman700_flat_tracking_env_cfg(play: bool = False, lock_hands: bool = False) -> ManagerBasedRlEnvCfg:
    """Args:
      play: auto-reset, one env, every clip from its first frame (mjlab's ``play``).
      lock_hands: hands + wrists locked (33 joint DoF), muscles acting only on them removed (694 muscles).
    """
    cfg = make_tracking_env_cfg()
    cfg.scene.entities = {"robot": get_mshuman700_cfg(self_collisions=False, lock_hands=lock_hands)}
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.body_names = TRACKED_BODIES
    cfg.viewer.body_name = "pelvis"
    # MyoFullBody's four foot touch sensors: heel and toes per side, as the ground's net force on those bodies
    cfg.scene.sensors = (
        ContactSensorCfg(
            name=motion_cmd.touch_sensor_name,
            primary=ContactMatch(mode="body", pattern=FOOT_BODIES, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("force",),
            reduce="netforce",
        ),
    )
    cfg.scene.num_envs = 4096  # paper Table A.5
    if play:
        # mjlab's play loop expects auto-reset; play every clip from its first frame
        cfg.scene.num_envs = 1
        cfg.auto_reset = True
        cfg.episode_length_s = int(1e9)
        motion_cmd.sampling_mode = "start"
        cfg.terminations.pop("time_out")
    return cfg
