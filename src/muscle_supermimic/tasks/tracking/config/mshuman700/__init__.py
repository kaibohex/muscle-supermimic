from mjlab.tasks.registry import register_mjlab_task

from muscle_supermimic.rl.supermimic import SuperMimicRunner

from .env_cfgs import mshuman700_flat_tracking_env_cfg
from .rl_cfg import mshuman700_supermimic_runner_cfg

register_mjlab_task(
    task_id="Mjlab-SuperMimic-Flat-MS-Human-700",
    env_cfg=mshuman700_flat_tracking_env_cfg(),
    play_env_cfg=mshuman700_flat_tracking_env_cfg(play=True),
    rl_cfg=mshuman700_supermimic_runner_cfg(),
    runner_cls=SuperMimicRunner,
)

register_mjlab_task(
    task_id="Mjlab-SuperMimic-Flat-MS-Human-700-Locked-Hands",
    env_cfg=mshuman700_flat_tracking_env_cfg(lock_hands=True),
    play_env_cfg=mshuman700_flat_tracking_env_cfg(play=True, lock_hands=True),
    rl_cfg=mshuman700_supermimic_runner_cfg(),
    runner_cls=SuperMimicRunner,
)

# without the muscle length / velocity / force observations: no muscle geometry on either side (docs/mdp.md)
for _suffix, _lock in (("", False), ("-Locked-Hands", True)):
    register_mjlab_task(
        task_id=f"Mjlab-SuperMimic-Flat-MS-Human-700{_suffix}-No-Muscle-Obs",
        env_cfg=mshuman700_flat_tracking_env_cfg(lock_hands=_lock, muscle_obs=False),
        play_env_cfg=mshuman700_flat_tracking_env_cfg(play=True, lock_hands=_lock, muscle_obs=False),
        rl_cfg=mshuman700_supermimic_runner_cfg(),
        runner_cls=SuperMimicRunner,
    )
