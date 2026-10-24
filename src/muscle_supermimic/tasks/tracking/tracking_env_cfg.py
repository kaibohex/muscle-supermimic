"""SuperMimic motion-tracking task configuration (robot-agnostic part).

Re-implements the MDP of SuperMimic (Liao et al., CoRL 2026, Appendix A) as reproduced by
supermimic_vibed (github.com/zita-ch/supermimic_vibed) on mjlab terms: BeyondMimic tracking reward
without the self-collision term, joint-limit weight -2, termination |e_z| > 0.25 m or e_tilt > 0.8,
proprioception with a 10-frame history and K = 10 future reference frames as the goal.

Two muscle-specific additions, taken from MuscleMimic (where SuperMimic trains stably on a muscle
model): the policy also observes the current muscle activations, and episodes also end on MuscleMimic's
deviation checks (mean root-relative body position, root position and root orientation).

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


def make_tracking_env_cfg() -> ManagerBasedRlEnvCfg:
    """Create the base tracking task configuration."""

    ##
    # Observations
    ##

    # supermimic: o_t = (q - q0, qd, IMU w, projected gravity, a_{t-1}, ref root ori), o_{t-9:t} to the decoder
    hist = {"params": CMD, "history_length": 10}
    proprio_terms = {
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos, **hist),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel, **hist),
        "root_ang_vel_b": ObservationTermCfg(func=mdp.root_ang_vel_b, **hist),
        "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity, **hist),
        "last_action": ObservationTermCfg(func=mdp.last_action, **hist),
        "motion_ref_ori_b": ObservationTermCfg(func=mdp.motion_ref_ori_b, **hist),
        # muscle-specific: the current activations (as MuscleMimic observes), no history
        "muscle_act": ObservationTermCfg(func=mdp.muscle_act, params=CMD),
    }
    # supermimic: g_t = K future reference frames, to the motion encoder
    goal_terms = {"motion_goal": ObservationTermCfg(func=mdp.motion_goal, params=CMD)}
    observations = {
        "policy": ObservationGroupCfg(terms=proprio_terms, concatenate_terms=True, enable_corruption=False),
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
            future_frames=10,
            sampling_mode="return",
        )
    }

    ##
    # Rewards
    ##

    rewards: dict[str, RewardTermCfg] = {
        "motion_root_pos": RewardTermCfg(func=mdp.motion_root_pos_exp, weight=0.5, params={**CMD, "std": 0.3}),
        "motion_root_ori": RewardTermCfg(func=mdp.motion_root_ori_exp, weight=0.5, params={**CMD, "std": 0.4}),
        "motion_body_pos": RewardTermCfg(func=mdp.motion_body_pos_exp, weight=1.0, params={**CMD, "std": 0.3}),
        "motion_body_ori": RewardTermCfg(func=mdp.motion_body_ori_exp, weight=1.0, params={**CMD, "std": 0.4}),
        "motion_body_lin_vel": RewardTermCfg(func=mdp.motion_body_lin_vel_exp, weight=1.0, params={**CMD, "std": 1.0}),
        "motion_body_ang_vel": RewardTermCfg(func=mdp.motion_body_ang_vel_exp, weight=1.0, params={**CMD, "std": 3.14}),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1, params=CMD),
        "joint_limit": RewardTermCfg(func=mdp.joint_pos_limits, weight=-2.0, params={**CMD, "soft_limit_factor": 0.9}),
    }

    ##
    # Terminations
    ##

    terminations: dict[str, TerminationTermCfg] = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "motion_end": TerminationTermCfg(func=mdp.motion_end, params=CMD, time_out=True),
        "root_height": TerminationTermCfg(func=mdp.bad_root_height, params={**CMD, "threshold": 0.25, "soft_width": 0.02}),
        "root_tilt": TerminationTermCfg(func=mdp.bad_root_tilt, params={**CMD, "threshold": 0.8, "soft_width": 0.04}),
        # MuscleMimic's deviation checks with supermusclemimic's thresholds (1 m, 1 m, 1 rad; soft width 5 % of each)
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
            mujoco=MujocoCfg(timestep=0.002, iterations=10, ls_iterations=20),
        ),
        decimation=10,  # 50 Hz control
        episode_length_s=10.0,
        # the SuperMimic runner resets done envs itself, after it has recorded their terminal state
        auto_reset=False,
    )
