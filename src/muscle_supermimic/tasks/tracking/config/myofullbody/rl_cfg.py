"""SuperMimic runner configuration for the MyoFullBody tracking task (paper Table A.5 defaults)."""

from muscle_supermimic.rl.supermimic import SuperMimicRunnerCfg


def myofullbody_supermimic_runner_cfg() -> SuperMimicRunnerCfg:
    return SuperMimicRunnerCfg(experiment_name="myofullbody_tracking", wandb_project="muscle_supermimic")
