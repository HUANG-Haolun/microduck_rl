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
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.robot.jduck_constants import (
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
JduckRlCfg.experiment_name = "jduck_velocity"
JduckRlCfg.run_name = "jduck_velocity"
