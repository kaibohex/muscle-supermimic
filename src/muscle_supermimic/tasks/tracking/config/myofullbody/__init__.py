from mjlab.tasks.registry import register_mjlab_task

from muscle_supermimic.rl.supermimic import SuperMimicRunner

from .env_cfgs import myofullbody_flat_tracking_env_cfg
from .rl_cfg import myofullbody_supermimic_runner_cfg

register_mjlab_task(
    task_id="Mjlab-SuperMimic-Flat-MyoFullBody",
    env_cfg=myofullbody_flat_tracking_env_cfg(),
    play_env_cfg=myofullbody_flat_tracking_env_cfg(play=True),
    rl_cfg=myofullbody_supermimic_runner_cfg(),
    runner_cls=SuperMimicRunner,
)
