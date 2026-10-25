# MDP

The task follows **MuscleMimic**: its 100 Hz control, observation, goal, MimicReward and deviation
terminations. Every term is an analytic function of the state, evaluated identically on simulator and
imagined states.

## One MDP, two sides

Every observation / reward / termination term of the env is a thin manager wrapper around an analytic
function of a state tuple `(p, R, v, w, q, qd, act)`, attached as `term.func.analytic`. The simulator
labels transitions through the managers; the runner evaluates the same functions, with the env cfg's
weights, stds and thresholds, on imagined states. The tests check that both give the same rewards,
observation history and goal.

## Terms

MuscleMimic's 17 mimic sites are the frames of 17 tracked bodies, pelvis first (the main site): on
MyoFullBody the sites' own bodies (pelvis, lumbar1, head, humerus, ulna, lunate, femur, tibia, talus,
toes), on MS-Human-700 bodies whose origins sit at the joints the sites mark (pelvis, femur, tibia, calcn,
toes x2, thoracic12, head_neck, humerus, ulna, hand x2).

* **Control**: 100 Hz (2 ms x 5 substeps), as MyoFullBody.
* **Observation** (decoder, no history): free-joint height and quaternion, every joint position, the
  free joint's qvel, every joint velocity, muscle length / velocity / force / excitation / activation,
  four foot touch values (ground force on heel and toes), and the root-relative site positions, rotation
  vectors and 6D velocities (GoalTrajMimic's state-dependent part).
* **Goal** (encoder): GoalTrajMimic's concise lookahead -- the reference's root-relative site positions
  at the current frame and 4 frames 0.2 s apart, with root position and qvel deltas -- and the motion phase.
* **Reward**: MimicReward with the full-body weights (qpos 0.1, qvel 0.1, root pos 0.1, root vel 0.1,
  site rpos 0.6, site rquat 0.01, site rvel 0.1 + 0.1; kernels 10 / 2 / 10 / 10 / 100 / 10 / 0.1), times a
  reward scale of 0.1; no penalties (all off in MuscleMimic's full-body configuration).
* **Termination**: mean root-relative site deviation > 1 m, root position deviation > 1 m, root
  orientation deviation > 1 rad (sigmoid-softened with 5 % widths in imagination); time-out at the clip
  end, bootstrapped (and at MuscleMimic's horizon of 30000 steps, 300 s, which no clip reaches).
* **Randomization**: none for now -- no friction / encoder offset / CoM domain randomization (the sim
  context ξ is removed from the world model and critic), no observation noise, no reset noise. Exploration
  noise on the actions and the sampling of reset frames are part of the algorithm and stay.

## Muscles on imagined states

Muscle length, velocity and force on imagined states come from a torch replica of MuJoCo's spatial
tendons (site-site segments, sphere / cylinder wraps with side sites and inside wraps) and Hill muscle
curves, the passive part on `biasprm` (`mdp/muscle_geometry.py`, equal to MuJoCo's `actuator_length /
velocity / force` to 1e-9 / 1e-8 / 1e-6; where a straight segment grazes a wrap surface, the wrap switches
on or off and MuJoCo, MuJoCo-Warp and the replica differ by microns). Activations are integrated
analytically with MuJoCo's own substeps (they match the simulator to float precision). Foot touch, which
depends on contacts, is predicted by an extra world-model head.

## Why MuscleMimic's MDP

With supermimic's own MDP (BeyondMimic reward with action-rate and joint-limit penalties, height/tilt
terminations, 50 Hz, 10-frame proprioceptive history) every MS-Human-700 run lost its critic 200-400
iterations after policy updates started. The penalties, summed over 694 muscles (the G1 humanoid has 29
actuators) and over the joints a collapsing body pins at their limits, were as large as the tracking
reward, so staying alive was worth nothing (imagined survival never rose). With MuscleMimic's non-negative,
penalty-free reward SuperMimic trains stably on a muscle model.
