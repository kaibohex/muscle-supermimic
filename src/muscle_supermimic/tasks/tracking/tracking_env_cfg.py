"""SuperMimic motion-tracking task configuration (robot-agnostic part).

SuperMimic (Liao et al., CoRL 2026) trained on MuscleMimic's MDP: 100 Hz control (2 ms x 5),
MuscleMimic's observation (full state incl. muscle length / velocity / force / excitation / activation
and foot touch, no history), its concise goal (5 reference frames 0.2 s apart and the motion phase), its
MimicReward with the full-body weights (no penalties), scaled by 0.1, and its deviation terminations
(mean root-relative site position, root position, root orientation; 1 m / 1 m / 1 rad).

No randomization of any kind: no domain randomization events, no observation noise, no reset noise.
Robot-specific settings are in ``config/``.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from muscle_supermimic.tasks.tracking import mdp

CMD = {"command_name": "motion"}
TIMESTEP, DECIMATION = 0.002, 5
REWARD_SCALE = 0.1


def make_tracking_env_cfg() -> ManagerBasedRlEnvCfg:
    """Create the base tracking task configuration."""

    ##
    # Observations
    ##

    # MuscleMimic: the decoder sees the full state and the state-dependent part of the goal (no history)
    decoder = ("root_pos", "joint_pos", "root_vel", "joint_vel", "muscle_length", "muscle_velocity", "muscle_force",
               "muscle_excitation", "muscle_activation", "touch", "site_rpos", "site_rangles", "site_rvel")  # fmt: skip
    decoder_terms = {name: ObservationTermCfg(func=getattr(mdp, name), params=CMD) for name in decoder}
    # the encoder sees the reference-only part of the goal
    goal_terms = {"motion_goal": ObservationTermCfg(func=mdp.motion_goal, params=CMD)}
    observations = {
        "policy": ObservationGroupCfg(terms=decoder_terms, concatenate_terms=True, enable_corruption=False),
        "goal": ObservationGroupCfg(terms=goal_terms, concatenate_terms=True, enable_corruption=False),
    }

    ##
    # Actions
    ##

    actions: dict[str, ActionTermCfg] = {"muscles": mdp.MuscleExcitationActionCfg(entity_name="robot")}

    ##
    # Commands
    ##

    commands: dict[str, CommandTermCfg] = {
        "motion": mdp.MotionCommandCfg(
            entity_name="robot",
            motion_file="",  # set on the CLI: --env.commands.motion.motion-file (play: --motion-file)
            body_names=(),  # set per robot
            sampling_mode="return",
        )
    }

    ##
    # Rewards
    ##

    # MimicReward: weight (w_sum) and kernel (w_exp) per component, times a reward scale of 0.1;
    # mjlab multiplies the weights by the step dt
    scale = REWARD_SCALE / (TIMESTEP * DECIMATION)
    mimic = {"qpos": (0.1, 10.0), "qvel": (0.1, 2.0), "root_pos": (0.1, 10.0), "root_vel": (0.1, 10.0), "rpos": (0.6, 100.0),
             "rquat": (0.01, 10.0), "rvel_rot": (0.1, 0.1), "rvel_lin": (0.1, 0.1)}  # fmt: skip
    rewards: dict[str, RewardTermCfg] = {
        f"mimic_{k}": RewardTermCfg(func=getattr(mdp, f"mimic_{k}"), weight=w * scale, params={**CMD, "w_exp": e})
        for k, (w, e) in mimic.items()
    }

    ##
    # Terminations
    ##

    terminations: dict[str, TerminationTermCfg] = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "motion_end": TerminationTermCfg(func=mdp.motion_end, params=CMD, time_out=True),
        "body_pos_dev": TerminationTermCfg(func=mdp.bad_body_pos_dev, params={**CMD, "threshold": 1.0, "soft_width": 0.05}),
        "root_pos_dev": TerminationTermCfg(func=mdp.bad_root_pos_dev, params={**CMD, "threshold": 1.0, "soft_width": 0.05}),
        "root_ori_dev": TerminationTermCfg(func=mdp.bad_root_ori_dev, params={**CMD, "threshold": 1.0, "soft_width": 0.05}),
    }

    ##
    # Assemble and return
    ##

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1),
        observations=observations,
        actions=actions,
        commands=commands,
        events={},  # no randomization; the motion command writes the full reset state
        rewards=rewards,
        terminations=terminations,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="",  # set per robot
            distance=3.0,
            fovy=55.0,
            elevation=-5.0,
            azimuth=120.0,
        ),
        sim=SimulationCfg(
            nconmax=48,
            njmax=320,
            # MuscleMimic's solver settings; as fast as mjlab's implicitfast at 10 / 20 on these models
            mujoco=MujocoCfg(timestep=TIMESTEP, integrator="euler", iterations=100, ls_iterations=50),
        ),
        decimation=DECIMATION,  # 100 Hz control, as MyoFullBody
        episode_length_s=300.0,  # MuscleMimic's horizon (30000 steps): episodes end at the clip's last frame
        # the SuperMimic runner resets done envs itself, after it has recorded their terminal state
        auto_reset=False,
    )
