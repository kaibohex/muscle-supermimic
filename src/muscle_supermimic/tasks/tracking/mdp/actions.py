"""Muscle excitation action: one excitation in [0, 1] per muscle, in actuator (= activation) order.

Muscles and their tendons are declared in different orders upstream, so the term resolves each
actuator's tendon by name and writes the excitation as that tendon's effort target, which mjlab's
``XmlActuator`` copies into the muscle's ``ctrl``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mjlab.envs.mdp.actions import TendonEffortActionCfg
from mjlab.envs.mdp.actions.actions import TendonEffortAction

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class MuscleExcitationAction(TendonEffortAction):
    def _find_targets(self, cfg: MuscleExcitationActionCfg) -> tuple[list[int], list[str]]:
        tendon_names = list(self._entity.tendon_names)
        names = [act.target.split("/")[-1] for act in self._entity.spec.actuators]  # as Entity.tendon_names
        return [tendon_names.index(n) for n in names], names


@dataclass(kw_only=True)
class MuscleExcitationActionCfg(TendonEffortActionCfg):
    actuator_names: tuple[str, ...] = (".*",)  # resolved from the actuators, see MuscleExcitationAction
    clip: dict[str, tuple] | None = field(default_factory=lambda: {".*": (0.0, 1.0)})

    def build(self, env: ManagerBasedRlEnv) -> MuscleExcitationAction:
        return MuscleExcitationAction(self, env)
