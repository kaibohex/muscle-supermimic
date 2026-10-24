"""SuperMimic: world-model BPTT motion tracking (Liao et al., CoRL 2026), as an mjlab runner."""

from .config import SuperMimicRunnerCfg
from .runner import SuperMimicRunner

__all__ = ["SuperMimicRunner", "SuperMimicRunnerCfg"]
