"""jduck BallKick: Microduck's ball-kick recipe on jduck.

Builds `make_microduck_ball_kick_env_cfg` (blind actor, critic sees the ball; support-foot
reward; linear-capped ball speed reward with overshoot penalty; ball ±1.5 cm DR) and swaps
in what is jduck-specific:
  - Robot: JDUCK_GROUNDCONTACT_ROBOT_CFG. Only the foot boxes (and trunk/hip/knee/head
    meshes) collide; the ball can hit the feet and the lower legs' ground meshes.
  - Ball placement from jduck's feet at the old HOME (measured): foot boxes 9 cm long,
    centres at y = ±0.064, toe tip at x = +0.032 from the trunk origin. Ball centre
    (0.09, ±0.064): toe → ball surface 2.3 cm (0.8 cm at worst with ±1.5 cm noise).
    Microduck: (0.09, ±0.042), toe at 0.034.
  - Standing height 0.133 (old HOME), standing spawn z around it.
  - UBT-12HC firmware replica + its DR instead of BAM; robot-style joint obs (ADC q,
    Δq qd, 1-step lag) like every jduck task. Default firmware JDUCK_FW=8,0,0.
  - Head bodies → jduck's.

Kick foot: JDUCK_KICK_FOOT=right|left (default right). Train each foot as its own run.
"""

import math
import os
from copy import deepcopy

from mjlab.managers import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.robot.jduck_constants import (
    JDUCK_ACTUATOR,
    JDUCK_FW_TAG,
    JDUCK_GROUNDCONTACT_ROBOT_CFG,
    JDUCK_HOME,
    JDUCK_HOME_TRUNK_Z,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_ball_kick_env_cfg as mk
from mjlab_microduck.tasks.jduck_sitstand_env_cfg import JDUCK_STAND_Z
from mjlab_microduck.tasks.jduck_velocity_env_cfg import (
    JDUCK_FRICTION_SCALE,
    JDUCK_HEAD_BODY_NAMES,
    JDUCK_KP_SCALE,
    JDUCK_OBS_ADC,
    JDUCK_OBS_LAG,
    JDUCK_SPEED_SCALE,
    JDUCK_TORQUE_SCALE,
    joint_pos_adc,
    joint_vel_adc,
    randomize_ubt12hc,
)

JDUCK_KICK_FOOT = os.environ.get("JDUCK_KICK_FOOT", "right")
assert JDUCK_KICK_FOOT in ("right", "left")
JDUCK_BALL_OFFSET_X = 0.09
JDUCK_BALL_OFFSET_ABS_Y = 0.064


def make_jduck_ball_kick_env_cfg(play: bool = False):
    if JDUCK_HOME != "old":
        import warnings
        warnings.warn("jduck ball kick is set up around the old HOME; set JDUCK_HOME=old to train it")
    foot = JDUCK_KICK_FOOT
    cfg = mk.make_microduck_ball_kick_env_cfg(play=play, kick_foot=foot)
    support = "left" if foot == "right" else "right"

    # ── robot + ball ──────────────────────────────────────────────────────────
    cfg.scene.entities = {"robot": JDUCK_GROUNDCONTACT_ROBOT_CFG, "ball": cfg.scene.entities["ball"]}
    sensors = []
    for s in cfg.scene.sensors:
        if s.name == "support_foot_ground_contact":
            s = deepcopy(s)
            s.primary = ContactMatch(mode="geom", pattern=rf"^{support}_foot_collision$", entity="robot")
        sensors.append(s)
    cfg.scene.sensors = tuple(sensors)

    # ── heights / spawns ──────────────────────────────────────────────────────
    cfg.rewards["height_stand"].params["target_height"] = JDUCK_STAND_Z
    z = JDUCK_HOME_TRUNK_Z
    gs = cfg.events["set_ground_state"].params
    gs["standing_z_min"], gs["standing_z_max"] = z + 0.002, z + 0.010
    cfg.events["reset_base"].params["pose_range"]["z"] = (z + 0.002, z + 0.010)
    oy = -JDUCK_BALL_OFFSET_ABS_Y if foot == "right" else JDUCK_BALL_OFFSET_ABS_Y
    cfg.events["reset_ball"].params["offset"] = (JDUCK_BALL_OFFSET_X, oy)
    # reset_ball must still run after set_ground_state (dict order): re-insert it last
    rb = cfg.events.pop("reset_ball")
    cfg.events["reset_ball"] = rb

    # ── head ──────────────────────────────────────────────────────────────────
    if "randomize_head_com" in cfg.events:
        cfg.events["randomize_head_com"].params["asset_cfg"] = SceneEntityCfg("robot", body_names=JDUCK_HEAD_BODY_NAMES)

    # ── actuator: UBT-12HC firmware replica, not BAM ──────────────────────────
    for name in ("randomize_joint_friction", "randomize_motor_gains"):
        cfg.events.pop(name, None)
    if JDUCK_ACTUATOR == "ubt12hc":
        cfg.events["randomize_ubt12hc"] = EventTermCfg(
            func=randomize_ubt12hc, mode="reset",
            params={"torque_range": JDUCK_TORQUE_SCALE, "speed_range": JDUCK_SPEED_SCALE,
                    "kp_range": JDUCK_KP_SCALE})
    if JDUCK_FRICTION_SCALE:
        cfg.events["randomize_jduck_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_dof_field_scaled, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",)),
                    "field": "dof_frictionloss", "scale_range": JDUCK_FRICTION_SCALE})

    # ── obs as the robot reads them ───────────────────────────────────────────
    if JDUCK_OBS_ADC:
        a = cfg.observations["actor"].terms
        a["joint_pos"].func = joint_pos_adc
        a["joint_pos"].noise = None
        a["joint_pos"].params.pop("biased", None)
        a["joint_vel"].func = joint_vel_adc
        a["joint_vel"].noise = None
        for t in ("joint_pos", "joint_vel"):
            a[t].delay_min_lag = JDUCK_OBS_LAG
            a[t].delay_max_lag = JDUCK_OBS_LAG
            a[t].delay_update_period = 0
    return cfg


JduckBallKickRlCfg = deepcopy(mk.MicroduckBallKickRlCfg)
JduckBallKickRlCfg.experiment_name = f"jduck_ball_kick_{JDUCK_KICK_FOOT}_{JDUCK_ACTUATOR}"
if JDUCK_ACTUATOR == "ubt12hc":
    JduckBallKickRlCfg.experiment_name += f"_{JDUCK_FW_TAG}"
JduckBallKickRlCfg.experiment_name += f"_home{JDUCK_HOME}"
if JDUCK_OBS_ADC:
    JduckBallKickRlCfg.experiment_name += "_adcobs" + (f"_lag{JDUCK_OBS_LAG}" if JDUCK_OBS_LAG else "")
if JDUCK_KP_SCALE:
    JduckBallKickRlCfg.experiment_name += f"_kpdr{JDUCK_KP_SCALE[0]:g}-{JDUCK_KP_SCALE[1]:g}"
JduckBallKickRlCfg.run_name = JduckBallKickRlCfg.experiment_name
JduckBallKickRlCfg.max_iterations = 5000


# ── Aim and power (jbk2) ─────────────────────────────────────────────────────
# jbk1 kicked 100 % but 17° off-axis at 1.5 m/s (target 1.0): ball_forward_velocity
# only counts the forward component (17° off still pays cos 17° = 96 %), and the
# overshoot penalty (−4, also forward-only) is cheap next to the longer roll a harder
# kick buys. JDUCK_KICK_LAT_W charges the ball's speed ACROSS the kick direction
# (1 m/s at 17° → 0.29 m/s lateral); JDUCK_KICK_OVER_W replaces the overshoot weight.
# Both stay below the forward reward's 12 so "kick roughly right" still beats "don't kick".
JDUCK_KICK_LAT_W = float(os.environ.get("JDUCK_KICK_LAT_W", "0"))
JDUCK_KICK_OVER_W = float(os.environ.get("JDUCK_KICK_OVER_W", "0"))


def ball_lateral_speed(env, asset_name: str = "ball", max_speed: float = 5.0):
    import torch

    ball = env.scene[asset_name]
    v = ball.data.root_link_lin_vel_w[:, :2]
    d = microduck_mdp._ball_kick_dir(env)
    lat = v[:, 0] * d[:, 1] - v[:, 1] * d[:, 0]
    return torch.nan_to_num(lat.abs(), nan=0.0).clamp(0.0, max_speed)


_make_base = make_jduck_ball_kick_env_cfg


def make_jduck_ball_kick_env_cfg(play: bool = False):   # noqa: F811
    cfg = _make_base(play=play)
    if JDUCK_KICK_LAT_W > 0:
        from mjlab.managers import RewardTermCfg

        cfg.rewards["ball_lateral_speed"] = RewardTermCfg(
            func=ball_lateral_speed, weight=-JDUCK_KICK_LAT_W, params={"asset_name": "ball"})
    if JDUCK_KICK_OVER_W > 0:
        cfg.rewards["ball_speed_overshoot"].weight = -JDUCK_KICK_OVER_W
    return cfg


if JDUCK_KICK_LAT_W > 0:
    JduckBallKickRlCfg.experiment_name += f"_lat{JDUCK_KICK_LAT_W:g}"
if JDUCK_KICK_OVER_W > 0:
    JduckBallKickRlCfg.experiment_name += f"_over{JDUCK_KICK_OVER_W:g}"
JduckBallKickRlCfg.run_name = JduckBallKickRlCfg.experiment_name
