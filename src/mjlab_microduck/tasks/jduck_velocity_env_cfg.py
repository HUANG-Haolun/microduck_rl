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

import os
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import EventTermCfg, RewardTermCfg
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
JDUCK_SPEED_SCALE = (0.85, 1.15)

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


def randomize_ubt12hc(
    env,
    env_ids,
    torque_range: tuple[float, float],
    speed_range: tuple[float, float],
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

        act.set_scales(env_ids, torque_scale=u(*torque_range), speed_scale=u(*speed_range))
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
            params={"torque_range": JDUCK_TORQUE_SCALE, "speed_range": JDUCK_SPEED_SCALE},
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
JduckRlCfg.run_name = JduckRlCfg.experiment_name
