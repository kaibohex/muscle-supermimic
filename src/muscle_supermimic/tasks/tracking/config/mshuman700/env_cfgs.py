"""MS-Human-700 flat-ground tracking environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg

from muscle_supermimic.assets.robots import get_mshuman700_cfg
from muscle_supermimic.tasks.tracking.mdp import MotionCommandCfg
from muscle_supermimic.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

# SuperMimic's 14 G1 bodies mapped onto the human skeleton (pelvis first: it is the root/anchor), plus the
# toes: hips, knees, ankles, toes, the trunk above the lumbar/thoracic joints, head, shoulders, elbows, wrists.
TRACKED_BODIES: tuple[str, ...] = (
    "pelvis",
    "femur_r", "tibia_r", "calcn_r", "toes_r",
    "femur_l", "tibia_l", "calcn_l", "toes_l",
    "thoracic12", "head_neck",
    "humerus_r", "ulna_r", "hand_r",
    "humerus_l", "ulna_l", "hand_l",
)  # fmt: skip


def mshuman700_flat_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Args:
      play: auto-reset, one env, every clip from its first frame (mjlab's ``play``).
    """
    cfg = make_tracking_env_cfg()
    cfg.scene.entities = {"robot": get_mshuman700_cfg(self_collisions=False)}
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.body_names = TRACKED_BODIES
    cfg.viewer.body_name = "pelvis"
    cfg.scene.num_envs = 4096  # paper Table A.5
    if play:
        # mjlab's play loop expects auto-reset; play every clip from its first frame
        cfg.scene.num_envs = 1
        cfg.auto_reset = True
        cfg.episode_length_s = int(1e9)
        motion_cmd.sampling_mode = "start"
        cfg.terminations.pop("time_out")
    return cfg
