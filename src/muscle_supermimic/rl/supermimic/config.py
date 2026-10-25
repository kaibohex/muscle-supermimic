"""SuperMimic runner configuration (paper Table A.5 / supermimic_vibed defaults)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl import RslRlBaseRunnerCfg


@dataclass
class SuperMimicRunnerCfg(RslRlBaseRunnerCfg):
    class_name: str = "SuperMimicRunner"
    obs_groups: dict[str, tuple[str, ...]] = field(default_factory=lambda: {"policy": ("policy",), "goal": ("goal",)})
    """Observation groups: ``policy`` is the decoder's proprioception history, ``goal`` the encoder's input."""
    command_name: str = "motion"
    max_iterations: int = 320_000
    save_interval: int = 2000
    log_interval: int = 50
    logger: Literal["wandb", "tensorboard"] = "tensorboard"

    # data collection
    buffer_steps: int = 1000
    """Replay buffer length per env (the buffer holds buffer_steps x num_envs transitions)."""
    utd: float = 0.4
    """Updates per vectorized simulator step (2.5 sim steps per iteration)."""
    action_noise: float = 0.1
    """Exploration noise N(0, sigma^2) added to the action, i.e. the excitations, during collection, then clipped to
    [0, 1]. Noise on the logits would shrink to sigma * e(1 - e) at low excitations."""
    warmup_steps: int = 64
    """Simulator steps before any update."""

    # world model
    wm_hidden: tuple[int, ...] = (2048, 2048, 2048)
    wm_window: int = 8
    wm_batches: int = 10
    wm_batch: int = 2048
    wm_only_iters: int = 500
    """World-model-only iterations at the start."""
    touch_loss_weight: float = 1.0
    """Weight of the world model's foot-touch loss."""

    # policy and critic
    enc_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    dec_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    latent_dim: int = 32
    critic_hidden: tuple[int, ...] = (256, 128)
    init_excitation: float = 0.1
    """Initial policy output: every muscle at this excitation (sigmoid of the decoder's output bias)."""
    horizon: int = 32
    pi_batch: int = 4096
    gamma: float = 0.99
    lam: float = 0.95
    critic_epochs: int = 5
    critic_minibatches: int = 4
    use_critic: bool = True
    remat_imagination: bool = True
    """Recompute each imagined step in the backward pass: the muscle geometry of every step would otherwise stay in
    memory."""
    critic_target_alpha: float = 0.0
    """Target critic for the policy's bootstrap and the lambda-return targets, V' <- alpha V' + (1 - alpha) V
    after each critic update (as in SHAC). 0 uses the online critic, as SuperMimic does."""
    value_clip: bool = False
    """Clamp bootstrapped values to the largest attainable return, max step reward / (1 - gamma), so the policy
    cannot exploit critic overestimation beyond it. Off in SuperMimic."""
    state_grad_clip: float = 0.0
    """Opt-in, not in SuperMimic: at every imagined step, clip each trajectory's gradient with respect to what is
    carried to the next step (body state, activation, touch, last action) to this norm, measured on that trajectory's
    own objective (the loss averages pi_batch of them). It bounds the closed-loop growth of the BPTT gradient
    (observation -> policy -> activation -> world model -> observation) that lets a few imagined trajectories dominate
    an update (MS-Human-700: pre-clip policy gradient norms of 1e2-1e5, ~100 % from 1 % of the trajectories). 0: off."""

    # optimization
    lr: float = 2e-4
    lr_decay_iters: int = 250_000
    """Cosine decay to 0.03 x lr over this many iterations."""
    grad_clip: float = 1.0
    policy_grad_clip_value: float = 0.1
    amp: Literal["none", "bf16"] = "bf16"
    """Autocast dtype for MLP matmuls (state integration, FK and rewards stay fp32)."""
    compile: bool = True
    """torch.compile the world-model rollout, imagination step and heads (CUDA only)."""
