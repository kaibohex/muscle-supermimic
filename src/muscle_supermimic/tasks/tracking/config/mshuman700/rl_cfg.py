"""SuperMimic runner configuration for the MS-Human-700 tracking task (paper Table A.5 defaults)."""

from muscle_supermimic.rl.supermimic import SuperMimicRunnerCfg


def mshuman700_supermimic_runner_cfg() -> SuperMimicRunnerCfg:
    return SuperMimicRunnerCfg(experiment_name="mshuman700_tracking", wandb_project="muscle_supermimic")
