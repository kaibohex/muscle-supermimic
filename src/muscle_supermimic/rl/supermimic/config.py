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
    state_grad_clip: float = 100.0
    """Not in SuperMimic (where it is 0, off): at every imagined step, clip each trajectory's gradient with respect
    to what is carried to the next step (body state, activation, touch, last action) to this norm, measured on that
    trajectory's own objective (the loss averages pi_batch of them). It bounds the closed-loop growth of the BPTT
    gradient (observation -> policy -> activation -> world model -> observation) that lets a few imagined
    trajectories dominate an update (MS-Human-700: pre-clip policy gradient norms of 1e2-1e5, ~100 % from 1 % of the
    trajectories). 0: off."""

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

    # ---- BlinkMimic / InstantTrack options (github.com/fuyichen0725/blinkmimic). The defaults reproduce the runner
    # above; scripts/train_blink.sh sets the drop-in recipe (docs/training.md) ----
    pi_ema: float = 0.0
    """Polyak average of the policy weights and normalizers (decay per policy update; InstantTrack: 0.995). The
    average is the deployed policy: checkpoints store it as "pi" (what ms-eval and play read) and the live policy as
    "pi_live"; collection keeps using the live policy. 0: off."""
    pi_opt: Literal["adam", "muon"] = "adam"
    """Policy optimizer: Adam, or Muon (Jordan et al. 2024) on the hidden weight matrices of the encoder and decoder at
    Adam's update RMS (same lr; Liu et al. 2025) with Adam on the first / last layers and vectors (optim.py)."""
    wm_opt: Literal["adam", "muon"] = "adam"
    """World-model optimizer: Adam, or the same Muon / Adam split on the world-model MLP."""
    muon_momentum: float = 0.95
    """Momentum of Muon's (Nesterov) update on the hidden matrices, policy and world model (torch.optim.Muon's default)."""
    pi_beta1: float = 0.9
    """Adam beta1 of the policy (SHAC uses 0.7)."""
    fused_adam: bool = False
    """Fused Adam kernels (with no_sync also the device-side skip of a non-finite policy update)."""
    no_sync: bool = False
    """Remove the host-device synchronizations of the update loop: the normalizers' freeze test, the sampling of
    imagination starts (a stable sort instead of nonzero) and the finite-gradient test before the policy step (the
    gradient is zeroed and the step gated on the device: needs fused_adam). Same updates up to floating-point rounding."""
    pi_lr_mult: float = 1.0
    """Policy learning rate = lr x this (world model and critic keep lr)."""
    lr_decay_ret: float = 0.0
    """Progress-triggered decay of pi_lr_mult (0: off): once the logged training return (ep_ret_norm, every
    log_interval iterations) reaches this, the multiplier ramps linearly to 1 over lr_decay_len iterations."""
    lr_decay_len: int = 300
    lr_decay_by: int = 0
    """With lr_decay_ret: start the decay at this iteration at the latest, also if the training return has not reached
    lr_decay_ret (0: no deadline)."""
    empty_cache: bool = True
    """torch.cuda.empty_cache() after every policy update: hands the update's cached blocks to MuJoCo-Warp, which
    allocates outside torch's pool, at the price of a device synchronization and the allocator refilling its cache."""
    wm_structured: bool = False
    """Semi-implicit world model (BlinkMimic): the network predicts the velocity increments and position corrections
    on top of the drift the old and new velocities imply over the step's Euler substeps, instead of the position
    increments themselves (networks.WorldModel)."""
    wm_mirror: float = 0.0
    """Share of each world-model batch replaced by its left-right mirror image (BlinkMimic: 0.5), the physical symmetry
    of the body as data augmentation: states, activations, excitations and foot touch are mirrored with the maps of
    tasks/tracking/mdp/mirror.py, validated on the model at start-up. 0: off."""
    eprobe_share: float = 0.0
    """Evaluation-guided resets (BlinkMimic: 0.15): this share of the training resets goes to the eprobe_win_s before
    the places where the deployed policy fails. Every eprobe_every iterations from eprobe_start, a sweep runs it,
    deterministic and without resets, in a second evaluation-only env: one probe world per eprobe_tile_s of every
    clip, started a random lead in eprobe_lead_s earlier on the reference and run to its tile's end; the per-bin
    counts of first failures decay by eprobe_decay per sweep. 0: off. The sweep runs eagerly and costs about as much
    as that many simulator steps of that many worlds (printed per sweep)."""
    eprobe_tile_s: float = 6.0
    eprobe_lead_s: tuple[float, float] = (4.0, 9.0)
    eprobe_every: int = 125
    eprobe_start: int = 1000
    """First sweep: before it nearly every probe fails and the focus would be about uniform."""
    eprobe_win_s: float = 10.0
    eprobe_decay: float = 0.7
    cuda_graph: bool = False
    """Systems (BlinkMimic): capture one world-model update and one policy + critic update (sampling, forward,
    backward, optimizer, EMA) as CUDA graphs after three warm-up calls and replay them from then on. The same updates
    up to floating-point rounding, on another random stream. Needs no_sync, fused_adam, empty_cache off and a CUDA
    device; the learning rate is then a device tensor filled every iteration, and Adam keeps its step on the GPU."""
    collect_graph: bool = False
    """Systems (BlinkMimic's --collect_graph --reset_masked): one collection step (observation, policy and action
    noise, the simulator's substeps including MuJoCo-Warp's solver loop, reward, termination, buffer write, normalizer
    updates, bin statistics and the resets of ended episodes as fixed-size masked updates) runs through a thin
    simulator wrapper (collector.py) in place of mjlab's manager env and is replayed as one CUDA graph after three
    eager warm-up steps: no host synchronization in a step. The same transitions up to floating-point rounding, on
    another random stream. Needs no_sync and the motion command's bin_fix, no eprobe_share and no observation history;
    the replay buffer's write head lives on the device. On a CPU device the wrapper runs eagerly."""
    max_hours: float = 0.0
    """Stop after this much training wall-clock (the runner's clock: compilation, graph capture and checkpoints
    included, process start-up and data loading excluded), saving a final checkpoint. 0: no limit."""
