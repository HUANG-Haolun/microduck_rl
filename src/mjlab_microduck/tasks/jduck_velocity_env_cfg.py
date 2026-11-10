"""jduck velocity (walking) environment.

Builds the microduck velocity recipe, which keeps the 61D obs, DR, noise, delays
and curricula, and then swaps in the jduck robot and fixes the names/values that
are microduck-specific:

  - robot entity → JDUCK_WALK_ROBOT_CFG (placeholder PD servo, see jduck_constants)
  - head-CoM DR body list → jduck head bodies
  - reset height → jduck HOME trunk height
  - velocity command ranges scaled down. jduck has ~0.59 N·m/kg of servo torque per
    kg of body mass vs microduck's ~1.2, and its servos are slower (3 rad/s
    placeholder vs ~10).
  - head-pose command caps clipped to jduck's (asymmetric) URDF head ranges

Everything else (rewards, curricula, obs layout) is unchanged on purpose, so a
run is directly comparable to Mjlab-Velocity-Flat-MicroDuck.
"""

import math
import os
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks import mdp as microduck_mdp

from mjlab_microduck.robot.jduck_constants import (
    JDUCK_ACTUATOR,
    JDUCK_FW_TAG,
    JDUCK_HOME,
    JDUCK_HOME_TRUNK_Z,
    JDUCK_WALK_ROBOT_CFG,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

JDUCK_HEAD_BODY_NAMES = (
    "neck_pitch_link",
    "head_pitch_link",
    "head_yaw_link",
    "head_roll_link",
)

# Conservative first-try command ranges (microduck: lin x ±0.4, y ±0.3, ang ±1.0).
JDUCK_LIN_VEL_X = (-0.25, 0.25)
JDUCK_LIN_VEL_Y = (-0.15, 0.15)
JDUCK_ANG_VEL_Z = (-0.8, 0.8)
# Gaussian tracking widths, scaled ∝ command range: microduck's sqrt(0.1) at
# ±0.4 m/s → sqrt(0.1)·0.25/0.4, and sqrt(0.5) at ±1.0 rad/s → sqrt(0.5)·0.8.
JDUCK_LIN_STD = 0.198
JDUCK_ANG_STD = 0.566

# Final head-pose command caps (deltas from HOME=0), kept ~10% inside the URDF
# limits. The limits are asymmetric, so take the smaller side to stay symmetric.
# neck_pitch [-1.53, 1.75], head_pitch [-1.63, 1.60], head_yaw [-0.85, 0.68],
# head_roll [-0.33, 0.29].
JDUCK_HEAD_CAPS = (1.10, 1.10, 0.60, 0.26)

# UBT-12HC replica DR (per joint, per episode). Stall torque −20/+15 %: the
# settle test falls at −25 %, so this keeps training feasible while teaching
# tolerance. Re-centre on the bench-measured value once identified.
JDUCK_TORQUE_SCALE = (0.80, 1.15)
# JDUCK_TORQUE_DR=<lo>,<hi> overrides it (the face-down/up get-up needs ≥ ~0.96 N·m:
# ~/jduck_ops/getup_search.py finds none at 0.77). Adds `_tdr<lo>-<hi>` to the name.
_tdr = os.environ.get("JDUCK_TORQUE_DR", "")
if _tdr:
    JDUCK_TORQUE_SCALE = tuple(float(x) for x in _tdr.split(","))
# Gearbox friction spread (JDUCK_FRICTION_DR=<lo>,<hi> multiplier on JDUCK_FRICTION).
# The inherited microduck friction DR only scales the BAM actuator → no-op on jduck.
_fr = os.environ.get("JDUCK_FRICTION_DR", "")
JDUCK_FRICTION_SCALE = tuple(float(x) for x in _fr.split(",")) if _fr else None
JDUCK_SPEED_SCALE = (0.85, 1.15)
# Firmware KP spread (JDUCK_KP_DR=<lo>,<hi> as a multiplier on K, default off).
# e.g. 0.75,1.25 → K=16 trains on K∈[12, 20]. Adds `_kpdr<lo>-<hi>` to the name.
_kp = os.environ.get("JDUCK_KP_DR", "")
JDUCK_KP_SCALE = tuple(float(x) for x in _kp.split(",")) if _kp else None

# Standing-still penalties (JDUCK_STAND_PENALTY=0 disables). The ubt12hc runs
# step in place at zero command (4–6 lift-offs/s in scripts/jduck_eval.py), while
# the PD run (v0) stands still. The firmware law (deadband + FF kick) makes it
# cheaper to balance by stepping than by holding. Same recipe as air_time but
# with the sign flipped: air_time pays for stepping while walking, these charge
# for stepping/jitter while standing. Both functions already exist in mdp.py.
#   foot_step_penalty_when_standing: 1 when standing, not pushed (body < 0.2 m/s)
#     and a foot just lifted. −1.0 ≈ the size of air_time (~1) for a robot that
#     keeps stepping, and it doesn't fire during push recovery.
#   joint_vel_l2_when_standing: Σ leg qd² while standing. Stepping in place is
#     O(10–50), so −0.01 is a noticeable but not dominant push.
#
# v1 got hacked: foot_step_penalty_when_standing only fires for an air phase
# > 50 ms. final4000 stood with ~20 lift-offs/s of ~15 ms each (5 mm high,
# unpenalized), which felt worse to drive than plain stepping.
#
# v2 (default) has no threshold to slip under:
#   feet_contact_when_standing: +per-step fraction of feet on the ground while
#     standing. Every 5 ms of air costs, however short.
#   stillness_at_zero_command: exp(-|v_xy|²/0.1²) while standing → don't drift.
#   leg_joint_vel_l2_when_standing: kept from v1 (continuous, no loophole).
# JDUCK_STAND=v1 reproduces the old runs (_standpen), 0 disables (also accepts the
# old JDUCK_STAND_PENALTY=0).
JDUCK_STAND = os.environ.get("JDUCK_STAND", "v2")
if os.environ.get("JDUCK_STAND_PENALTY") == "0":
    JDUCK_STAND = "0"
assert JDUCK_STAND in ("0", "v1", "v2"), JDUCK_STAND
JDUCK_STAND_PENALTY = JDUCK_STAND != "0"
JDUCK_STAND_STEP_W = -1.0
JDUCK_STAND_JOINT_VEL_W = -0.01
JDUCK_STAND_CONTACT_W = 1.0     # ≈ air_time's walking payoff, but for both feet down
JDUCK_STAND_STILL_W = 1.0

# Zero-command action-rate penalty (JDUCK_STAND_ACT_RATE=<w>, default 0 = off, so
# older runs rebuild unchanged). ff0db0 stood "still" by the stand2 metrics but
# kept moving its targets ~2–4°/step at zero command, even in its own training
# env (and the jitter is in the policy: the same policy on PD jitters too, v0 on
# the firmware law does not). stand2 pays for feet down and no drift, not for
# quiet targets. Σ_joints (a_t − a_{t−1})² at that jitter is ~0.05, so w=20
# costs ~1 per step, the size of one stand2 term; small balance corrections
# (quadratic) stay cheap. Adds `_sar<w>` to the experiment name.
JDUCK_STAND_ACT_RATE_W = float(os.environ.get("JDUCK_STAND_ACT_RATE", "0"))
# sar20 (w=20 from iteration 0) never learned to stand: episodes 18 → 238 steps
# vs ~970 without it. Taxing the early random actions blocks balance discovery
# (AGENTS.md: introduce smoothness after the skill exists). So the weight is
# staged: 0 until JDUCK_SAR_START (iterations), then 1/4, 1/2, full, spaced
# JDUCK_SAR_RAMP iterations apart.
JDUCK_SAR_START = int(os.environ.get("JDUCK_SAR_START", "1500"))
JDUCK_SAR_RAMP = int(os.environ.get("JDUCK_SAR_RAMP", "500"))

# Positive "quiet stand" reward (JDUCK_STAND_QUIET=<w>, default 0 = off):
# standing · exp(−Σ(a_t − a_{t−1})² / JDUCK_STAND_QUIET_S2). The action-rate
# PENALTY has a cliff: −5 stood still on seed 1 but jittered (1°/step) on seed 2,
# and −7.5/−10 made standing net-negative, so the policy learned to fall (no
# termination cost). A bounded positive term can't make dying attractive.
# S2 = 0.005: ff0db0's jitter (Σ ≈ 0.04) earns e^−8 ≈ 0, ~0.1°/step-level
# corrections (Σ ≈ 1e−4) keep ~98 %. Adds `_quiet<w>` to the experiment name.
JDUCK_STAND_QUIET_W = float(os.environ.get("JDUCK_STAND_QUIET", "0"))
JDUCK_STAND_QUIET_S2 = float(os.environ.get("JDUCK_STAND_QUIET_S2", "0.005"))

# Termination penalty (JDUCK_TERM=<w>, default 0 = off): −w per fall (×dt like
# every term). Without it a fall just ends the episode, so once the zero-command
# terms make standing net-negative (sar ≥ 7.5) the policy learns to fall (r1, r4).
# w=200 ≈ 2 s of the full positive stack. Adds `_term<w>` to the experiment name.
JDUCK_TERM_W = float(os.environ.get("JDUCK_TERM", "0"))

# Hardware-like joint observations (JDUCK_OBS_ADC=1, default 0 = off). On the
# robot, jduck_run.py only has the servo ADC: q is quantized to 2π/1024 rad and qd
# is (q_t − q_{t−1}) / 20 ms of those quantized readings (steps of ~0.31 rad/s).
# Training fed the true q and qd (+ ±0.25 noise), and every policy trained that
# way (r2, r7, r8) jittered at zero command once run through jduck_run.py +
# sim_robot.py: 0.4–1.0°/step vs 0.02° in training. Isolated: finite-difference
# qd is the main cause (0.016° → 3.1°), quantized q a small one (→ 0.17°).
# With this on, the ACTOR sees exactly that; the critic keeps true q/qd.
# Adds `_adcobs` to the experiment name.
JDUCK_OBS_ADC = os.environ.get("JDUCK_OBS_ADC", "0") == "1"
# qd low-pass for the ADC obs (JDUCK_QD_EMA=<alpha>, default 1 = raw Δq/dt): the
# actor sees qd_f += alpha·(Δq/dt − qd_f), and jduck_run.py must apply the same
# filter (it reads it from the ONNX metadata key "jduck_qd_ema"). Measured on r9 in
# the hw chain: holding qd constant at zero command drops the jitter 10.9 → 0.16°/step
# while freezing gyro, gravity or q makes it fall, so the oscillation runs through
# the quantized-velocity input. Adds `_qdema<alpha>` to the experiment name.
JDUCK_QD_EMA = float(os.environ.get("JDUCK_QD_EMA", "1"))
# 1-policy-step lag on the ADC joint obs (JDUCK_OBS_LAG=1). jduck_run.py builds the obs
# from the replies to LAST tick's CMD 0x01, so q (and qd from it) is ~20 ms old on the
# robot, while training fed this tick's reading. Through the hw chain, policies that walk
# straight with fresh obs (p10: dx +4.4 m, yaw ≈ 0) veer and spin with the stale ones
# (dx −0.4…+1.6 m, yaw −50…−240°). Adds `_lag1`.
JDUCK_OBS_LAG = int(os.environ.get("JDUCK_OBS_LAG", "0"))

# CAPS action-smoothness regularization in PPO (tasks/ppo_caps.py). With
# JDUCK_OBS_ADC=1 the policy still jitters ~2°/step at zero command in its own
# env (r9): reward terms cannot make the policy function insensitive to sensor
# noise, so regularize the function directly.
#   JDUCK_CAPS_S  spatial weight  (‖π(o) − π(o+ε)‖², ε ~ sensor noise), 0 = off
#   JDUCK_CAPS_T  temporal weight (‖π(o_t) − π(o_t+1)‖²), 0 = off
#   JDUCK_CAPS_STAND  extra weight on zero-command samples (default 4)
# Adds `_capsS<s>T<t>[b<boost>]` to the experiment name.
JDUCK_CAPS_S = float(os.environ.get("JDUCK_CAPS_S", "0"))
JDUCK_CAPS_T = float(os.environ.get("JDUCK_CAPS_T", "0"))
JDUCK_CAPS_STAND = float(os.environ.get("JDUCK_CAPS_STAND", "4"))
# Weight on walking (nonzero command) samples, default 1. c1 (S=1, T=0.1, weight 1
# everywhere) cut zero-speed jitter 10° → 0.46°/step but fell 3.4× more in training
# (fell_over 0.86 vs 0.25, air_time 0.84 vs 1.04): smoothing the walking policy costs
# balance. JDUCK_CAPS_WALK=0 regularizes only zero-command samples. Adds `w<val>`.
JDUCK_CAPS_WALK = float(os.environ.get("JDUCK_CAPS_WALK", "1"))
# Standing samples are smoothed only while the trunk is calm (|gyro| < JDUCK_CAPS_CALM
# rad/s; 0 = always). Standing-only CAPS still fell 3× more than no CAPS (L1 0.78, L2
# 0.64 vs p8 0.13): a pushed robot at zero command needs fast corrections. Adds `c<val>`.
JDUCK_CAPS_CALM = float(os.environ.get("JDUCK_CAPS_CALM", "0"))
# CAPS off until iteration JDUCK_CAPS_START (default 0 = from the start). Smoothing a
# policy that cannot walk yet may be why CAPS runs fall more. Adds `s<it>` to the name.
JDUCK_CAPS_START = int(os.environ.get("JDUCK_CAPS_START", "0"))

# Shift every iteration-staged curriculum (action-rate weight, standing-env share,
# head/COM ranges, head-pose weight) later by JDUCK_CURR_DELAY iterations. The stock
# firmware only starts stepping around iteration 800 (O1), but the action-rate penalty
# and the standing share start ramping at 500. Adds `_cd<it>` to the name.
JDUCK_CURR_DELAY = int(os.environ.get("JDUCK_CURR_DELAY", "0"))

# Yaw-rate tracking on a moving average (JDUCK_ANG_EMA=<tau s>, default 0 = the
# stock instantaneous term). Measured on ff0db0_homeold walking at 0.2 m/s: mean
# wz −0.15 rad/s but instantaneous (cmd − wz)² ≈ 7.7, because the trunk twists
# ±~2.7 rad/s every step; plus roll/pitch rate² ≈ 0.6–1.1 that the stock term
# also charges. exp(−(z+xy)/0.32) ≈ 0 whether the robot turns or not, so turning
# had no gradient (every jduck run, v0 included, turned 0–35 % of the command).
# Tracking an EMA of wz lets the step twist average out and prices only the
# mean turn rate. Adds `_angema<tau>` to the experiment name.
JDUCK_ANG_EMA = float(os.environ.get("JDUCK_ANG_EMA", "0"))


def randomize_ubt12hc(
    env,
    env_ids,
    torque_range: tuple[float, float],
    speed_range: tuple[float, float],
    kp_range: tuple[float, float] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Per-episode stall-torque / no-load-speed scale + encoder phase for the
    UBT-12HC replica (non-accumulating: absolute scales are written every time).
    No-op on other actuators."""
    import torch

    from mjlab_microduck.actuator.ubt12hc import Ubt12hcActuator

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(env.device, dtype=torch.long)
    for act in env.scene[asset_cfg.name].actuators:
        if not isinstance(act, Ubt12hcActuator):
            continue
        n = act.torque_scale.shape[1]

        def u(lo, hi):
            # per-joint: servos differ from each other
            return torch.rand(len(env_ids), n, device=env.device) * (hi - lo) + lo

        act.set_scales(env_ids, torque_scale=u(*torque_range), speed_scale=u(*speed_range),
                       kp_scale=u(*kp_range) if kp_range else None)
        act.reset_encoder_phase(env_ids)


def feet_contact_when_standing(
    env,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    sensor_name: str = "feet_ground_contact",
):
    """Fraction of feet (0, 0.5, 1) on the ground, only while the twist command is ~0.

    Counted every step, so even a 5 ms hop off the ground costs. No air-time
    threshold to slip under (see the v1 note above).
    """
    import torch

    found = env.scene.sensors[sensor_name].data.found   # (N, 2) contact count per foot
    on_ground = (found > 0).float().mean(dim=-1)
    cmd = env.command_manager.get_command(command_name)
    standing = (torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2]) < command_threshold).float()
    return standing * on_ground


def leg_joint_vel_l2_when_standing(
    env,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Σ leg joint qd² while the twist command is ~0.

    Same as mdp.joint_vel_l2_when_standing, but it maps through _servo_joint_ids.
    That one hard-codes entity indices 0–4/9–13, and on jduck `passive_mouth` sits
    at index 9, so it would pick the mouth and drop the right ankle.
    """
    import torch

    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    standing = (torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2]) < command_threshold).float()
    servo = microduck_mdp._servo_joint_ids(env, asset)
    legs = [servo[i] for i in (*range(0, 5), *range(9, 14))]   # 14-servo layout → entity ids
    return standing * torch.sum(asset.data.joint_vel[:, legs] ** 2, dim=-1)


def action_rate_l2_when_standing(
    env,
    command_name: str = "twist",
    command_threshold: float = 0.01,
):
    """Σ (a_t − a_{t−1})² over all actions, only while the twist command is ~0."""
    import torch

    cmd = env.command_manager.get_command(command_name)
    standing = (torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2]) < command_threshold).float()
    da = env.action_manager.action - env.action_manager.prev_action
    return standing * torch.sum(da ** 2, dim=-1)


class track_yaw_rate_ema:
    """exp(−(cmd_wz − EMA(wz))² / std²): yaw-rate tracking on a moving average.

    EMA time constant `tau` (s). Reset to 0 at episode start; the command is
    held for several seconds, so the lag matters only right after a resample.
    """

    def __init__(self, cfg, env):
        import torch

        self.ema = torch.zeros(env.num_envs, device=env.device)
        self.alpha = min(1.0, env.step_dt / cfg.params["tau"])

    def reset(self, env_ids=None):
        if env_ids is None:
            self.ema[:] = 0.0
        else:
            self.ema[env_ids] = 0.0

    def __call__(self, env, std: float, tau: float, command_name: str = "twist",
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
        import torch

        del tau
        wz = env.scene[asset_cfg.name].data.root_link_ang_vel_b[:, 2]
        self.ema += self.alpha * (wz - self.ema)
        cmd = env.command_manager.get_command(command_name)
        return torch.exp(-torch.square(cmd[:, 2] - self.ema) / std**2)


def quiet_actions_when_standing(
    env,
    s2: float,
    command_name: str = "twist",
    command_threshold: float = 0.01,
):
    """exp(−Σ (a_t − a_{t−1})² / s2), only while the twist command is ~0."""
    import torch

    cmd = env.command_manager.get_command(command_name)
    standing = (torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2]) < command_threshold).float()
    da = env.action_manager.action - env.action_manager.prev_action
    return standing * torch.exp(-torch.sum(da ** 2, dim=-1) / s2)


class _AdcJointState:
    """Per-env ADC encoder model shared by the two actor obs terms below.

    Quantized q = floor(q_biased / k + phase) · k (phase: per-env per-joint random
    offset, resampled at reset, like the ADC's arbitrary zero), and qd = Δq / dt of
    those readings. Updated once per env step (cached by step counter).
    """

    K = 2 * math.pi / 1024

    def __init__(self, env, joint_ids):
        import torch

        n = len(joint_ids)
        self.ids = joint_ids
        self.phase = torch.rand(env.num_envs, n, device=env.device)
        self.prev = None
        self.q = None
        self.qd = torch.zeros(env.num_envs, n, device=env.device)
        self.step = -1
        self.fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def update(self, env):
        import torch

        step = env.common_step_counter
        if step == self.step and self.q is not None:
            return
        asset = env.scene["robot"]
        raw = asset.data.joint_pos_biased[:, self.ids]
        q = (torch.floor(raw / self.K + self.phase) - self.phase + 0.5) * self.K
        if self.prev is None:
            self.prev = q.clone()
        raw = (q - self.prev) / env.step_dt
        raw = torch.where(self.fresh[:, None], torch.zeros_like(raw), raw)
        qd_prev = torch.where(self.fresh[:, None], torch.zeros_like(raw), self.qd)
        qd = qd_prev + JDUCK_QD_EMA * (raw - qd_prev)
        self.fresh[:] = False
        self.prev, self.q, self.qd, self.step = q, q, qd, step

    def reset(self, env_ids):
        import torch

        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.phase.shape[0], device=self.phase.device)
        self.phase[env_ids] = torch.rand(len(env_ids), self.phase.shape[1], device=self.phase.device)
        self.fresh[env_ids] = True
        self.step = -1   # force a refresh so the reset envs get a new reading


_ADC_STATE = {}


def _adc_state(env, asset_cfg):
    key = id(env)
    if key not in _ADC_STATE:
        _ADC_STATE[key] = _AdcJointState(env, asset_cfg.joint_ids)
    return _ADC_STATE[key]


class joint_pos_adc:
    """Actor joint_pos as the robot reads it: ADC-quantized, relative to HOME."""

    def __init__(self, cfg, env):
        self.state = _adc_state(env, cfg.params["asset_cfg"])

    def reset(self, env_ids=None):
        self.state.reset(env_ids)

    def __call__(self, env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), biased: bool = True):
        del biased
        self.state.update(env)
        default = env.scene["robot"].data.default_joint_pos[:, asset_cfg.joint_ids]
        return self.state.q - default


class joint_vel_adc:
    """Actor joint_vel as jduck_run.py computes it: Δ(quantized q) / 20 ms."""

    def __init__(self, cfg, env):
        self.state = _adc_state(env, cfg.params["asset_cfg"])

    def reset(self, env_ids=None):
        self.state.reset(env_ids)

    def __call__(self, env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
        self.state.update(env)
        return self.state.qd


def _clip_ranges(ranges, caps):
    return tuple(
        (max(lo, -c), min(hi, c)) for (lo, hi), c in zip(ranges, caps)
    )


def make_jduck_velocity_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

    cfg.scene.entities = {"robot": JDUCK_WALK_ROBOT_CFG}

    if "randomize_head_com" in cfg.events:
        cfg.events["randomize_head_com"].params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=JDUCK_HEAD_BODY_NAMES
        )

    z = JDUCK_HOME_TRUNK_Z
    cfg.events["reset_base"].params["pose_range"]["z"] = (z + 0.002, z + 0.010)

    if JDUCK_ACTUATOR == "ubt12hc":
        # The replica has no kp/kd to randomize (the microduck KP/KD DR is off
        # anyway). Randomize what is actually uncertain: the stall torque (whether
        # the 48-PWM floor holds the stance hinges on it) and the motor speed.
        cfg.events["randomize_ubt12hc"] = EventTermCfg(
            func=randomize_ubt12hc,
            mode="reset",
            params={"torque_range": JDUCK_TORQUE_SCALE, "speed_range": JDUCK_SPEED_SCALE,
                    "kp_range": JDUCK_KP_SCALE},
        )

    if JDUCK_FRICTION_SCALE:
        cfg.events["randomize_jduck_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_dof_field_scaled,   # field expanded per env by expand_bam_friction_fields
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",)),
                "field": "dof_frictionloss",
                "scale_range": JDUCK_FRICTION_SCALE,
            },
        )

    twist = deepcopy(cfg.commands["twist"])
    twist.ranges.lin_vel_x = JDUCK_LIN_VEL_X
    twist.ranges.lin_vel_y = JDUCK_LIN_VEL_Y
    twist.ranges.ang_vel_z = JDUCK_ANG_VEL_Z
    cfg.commands["twist"] = twist

    # Tracking std must scale with the command range. With microduck's
    # std²=0.1 and jduck's ±0.25 m/s, STANDING STILL already earns ~83% of
    # track_linear_velocity. The first ubt12hc run (2026-11-03) found exactly
    # that: air_time ≈ 0, dx = 0 over 8 s at a 0.2 m/s command, while the logged
    # tracking reward looked healthy (1.46/2.0). Keep the stand-still payoff
    # near microduck's (~65% lin / ~60% ang) at jduck's smaller ranges.
    cfg.rewards["track_linear_velocity"].params["std"] = JDUCK_LIN_STD
    cfg.rewards["track_angular_velocity"].params["std"] = JDUCK_ANG_STD
    if JDUCK_ANG_EMA > 0:
        w = cfg.rewards["track_angular_velocity"].weight
        cfg.rewards["track_angular_velocity"] = RewardTermCfg(
            func=track_yaw_rate_ema,
            weight=w,
            params={"std": JDUCK_ANG_STD, "tau": JDUCK_ANG_EMA, "command_name": "twist"},
        )

    if JDUCK_STAND == "v1":
        cfg.rewards["stand_step_penalty"] = RewardTermCfg(
            func=microduck_mdp.foot_step_penalty_when_standing,
            weight=JDUCK_STAND_STEP_W,
            params={"command_threshold": 0.01, "body_vel_threshold": 0.2, "air_time_threshold": 0.05},
        )
    if JDUCK_STAND == "v2":
        cfg.rewards["stand_feet_contact"] = RewardTermCfg(
            func=feet_contact_when_standing,
            weight=JDUCK_STAND_CONTACT_W,
            params={"command_threshold": 0.01},
        )
        cfg.rewards["stand_still"] = RewardTermCfg(
            func=microduck_mdp.stillness_at_zero_command,
            weight=JDUCK_STAND_STILL_W,
            params={"command_threshold": 0.01, "vel_std": 0.1},
        )
    if JDUCK_STAND_PENALTY:
        cfg.rewards["stand_joint_vel"] = RewardTermCfg(
            func=leg_joint_vel_l2_when_standing,
            weight=JDUCK_STAND_JOINT_VEL_W,
            params={"command_threshold": 0.01},
        )

    if JDUCK_STAND_ACT_RATE_W > 0:
        cfg.rewards["stand_action_rate"] = RewardTermCfg(
            func=action_rate_l2_when_standing,
            weight=0.0 if JDUCK_SAR_START > 0 else -JDUCK_STAND_ACT_RATE_W,
            params={"command_threshold": 0.01},
        )
        if JDUCK_SAR_START > 0:
            steps = 24   # NUM_STEPS_PER_ENV: curriculum steps are env steps
            cfg.curriculum["stand_action_rate_weight"] = CurriculumTermCfg(
                func=microduck_mdp.reward_weight,
                params={
                    "reward_name": "stand_action_rate",
                    "weight_stages": [
                        {"step": 0, "weight": 0.0},
                        *({"step": (JDUCK_SAR_START + i * JDUCK_SAR_RAMP) * steps,
                           "weight": -JDUCK_STAND_ACT_RATE_W * f}
                          for i, f in enumerate((0.25, 0.5, 1.0))),
                    ],
                },
            )

    if JDUCK_TERM_W > 0:
        from mjlab.envs import mdp as mjlab_mdp

        cfg.rewards["termination"] = RewardTermCfg(func=mjlab_mdp.is_terminated, weight=-JDUCK_TERM_W)

    if JDUCK_STAND_QUIET_W > 0:
        cfg.rewards["stand_quiet"] = RewardTermCfg(
            func=quiet_actions_when_standing,
            weight=JDUCK_STAND_QUIET_W,
            params={"s2": JDUCK_STAND_QUIET_S2, "command_threshold": 0.01},
        )

    if JDUCK_OBS_ADC:
        a = cfg.observations["actor"].terms
        a["joint_pos"].func = joint_pos_adc
        a["joint_pos"].noise = None          # the quantization IS the noise
        a["joint_vel"].func = joint_vel_adc
        a["joint_vel"].noise = None
        a["joint_vel"].delay_min_lag = JDUCK_OBS_LAG   # Δq/dt is already the last 20 ms average
        a["joint_vel"].delay_max_lag = JDUCK_OBS_LAG
        a["joint_pos"].delay_min_lag = JDUCK_OBS_LAG
        a["joint_pos"].delay_max_lag = JDUCK_OBS_LAG

    if JDUCK_CURR_DELAY > 0:
        d = JDUCK_CURR_DELAY * 24     # env steps per iteration
        for name, term in cfg.curriculum.items():
            for key in ("weight_stages", "standing_stages", "range_stages", "stages"):
                st = term.params.get(key)
                if st:
                    term.params[key] = [{**s, "step": (s["step"] + d) if s["step"] > 0 else 0} for s in st]

    head = deepcopy(cfg.commands["head_pose"])
    head.ranges = _clip_ranges(head.ranges, JDUCK_HEAD_CAPS)
    cfg.commands["head_pose"] = head
    stages = cfg.curriculum["head_pose_range"].params["range_stages"]
    cfg.curriculum["head_pose_range"].params["range_stages"] = [
        {**s, "ranges": _clip_ranges(s["ranges"], JDUCK_HEAD_CAPS)} for s in stages
    ]

    return cfg


JduckRlCfg = deepcopy(MicroduckRlCfg)
JduckRlCfg.experiment_name = f"jduck_velocity_{JDUCK_ACTUATOR}"
if JDUCK_ACTUATOR == "ubt12hc":
    JduckRlCfg.experiment_name += f"_{JDUCK_FW_TAG}"
JduckRlCfg.experiment_name += f"_home{JDUCK_HOME}"
if JDUCK_STAND == "v1":
    JduckRlCfg.experiment_name += "_standpen"
elif JDUCK_STAND == "v2":
    JduckRlCfg.experiment_name += "_stand2"
if JDUCK_STAND_ACT_RATE_W > 0:
    JduckRlCfg.experiment_name += f"_sar{JDUCK_STAND_ACT_RATE_W:g}"
    if JDUCK_SAR_START > 0:
        JduckRlCfg.experiment_name += f"from{JDUCK_SAR_START}"
if JDUCK_STAND_QUIET_W > 0:
    JduckRlCfg.experiment_name += f"_quiet{JDUCK_STAND_QUIET_W:g}"
if JDUCK_STAND_QUIET_W > 0 and JDUCK_STAND_QUIET_S2 != 0.005:
    JduckRlCfg.experiment_name += f"s{JDUCK_STAND_QUIET_S2:g}"
if JDUCK_TERM_W > 0:
    JduckRlCfg.experiment_name += f"_term{JDUCK_TERM_W:g}"
if JDUCK_ANG_EMA > 0:
    JduckRlCfg.experiment_name += f"_angema{JDUCK_ANG_EMA:g}"
if JDUCK_OBS_ADC:
    JduckRlCfg.experiment_name += "_adcobs"
    if JDUCK_QD_EMA != 1:
        JduckRlCfg.experiment_name += f"_qdema{JDUCK_QD_EMA:g}"
    if JDUCK_OBS_LAG:
        JduckRlCfg.experiment_name += f"_lag{JDUCK_OBS_LAG}"
if os.environ.get("JDUCK_FRICTION"):
    JduckRlCfg.experiment_name += f"_fr{float(os.environ['JDUCK_FRICTION']):g}"
if JDUCK_FRICTION_SCALE:
    JduckRlCfg.experiment_name += f"_frdr{JDUCK_FRICTION_SCALE[0]:g}-{JDUCK_FRICTION_SCALE[1]:g}"
if JDUCK_KP_SCALE:
    JduckRlCfg.experiment_name += f"_kpdr{JDUCK_KP_SCALE[0]:g}-{JDUCK_KP_SCALE[1]:g}"
if os.environ.get("JDUCK_TORQUE_DR"):
    JduckRlCfg.experiment_name += f"_tdr{JDUCK_TORQUE_SCALE[0]:g}-{JDUCK_TORQUE_SCALE[1]:g}"
if JDUCK_CAPS_S > 0 or JDUCK_CAPS_T > 0:
    from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg as _PpoCfg
    from dataclasses import dataclass as _dc

    @_dc
    class _PpoCapsCfg(_PpoCfg):
        caps_spatial: float = 0.0
        caps_temporal: float = 0.0
        caps_stand_boost: float = 4.0
        caps_walk_weight: float = 1.0
        caps_calm: float = 0.0
        caps_start_iter: int = 0

    _a = JduckRlCfg.algorithm
    JduckRlCfg.algorithm = _PpoCapsCfg(**{**vars(_a), "class_name": "mjlab_microduck.tasks.ppo_caps:PPOCaps",
                                         "caps_spatial": JDUCK_CAPS_S, "caps_temporal": JDUCK_CAPS_T,
                                         "caps_stand_boost": JDUCK_CAPS_STAND, "caps_walk_weight": JDUCK_CAPS_WALK,
                                         "caps_calm": JDUCK_CAPS_CALM, "caps_start_iter": JDUCK_CAPS_START})
    JduckRlCfg.experiment_name += f"_capsS{JDUCK_CAPS_S:g}T{JDUCK_CAPS_T:g}"
    if JDUCK_CAPS_STAND != 4:
        JduckRlCfg.experiment_name += f"b{JDUCK_CAPS_STAND:g}"
    if JDUCK_CAPS_WALK != 1:
        JduckRlCfg.experiment_name += f"w{JDUCK_CAPS_WALK:g}"
    if JDUCK_CAPS_CALM > 0:
        JduckRlCfg.experiment_name += f"c{JDUCK_CAPS_CALM:g}"
    if JDUCK_CAPS_START > 0:
        JduckRlCfg.experiment_name += f"s{JDUCK_CAPS_START}"
if JDUCK_CURR_DELAY > 0:
    JduckRlCfg.experiment_name += f"_cd{JDUCK_CURR_DELAY}"
JduckRlCfg.run_name = JduckRlCfg.experiment_name
