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

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.robot.jduck_constants import (
    JDUCK_ACTUATOR,
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
JduckRlCfg.run_name = f"jduck_velocity_{JDUCK_ACTUATOR}"
