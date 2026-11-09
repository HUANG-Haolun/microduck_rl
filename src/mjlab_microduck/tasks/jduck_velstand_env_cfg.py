"""jduck VelStand: walk + stand + fall + get up in one policy, Microduck's recipe on jduck.

Builds microduck_velstand_env_cfg (warm start from a walk, fallen-gated behaviour cloning
toward a frozen stand-up teacher and a frozen walk anchor, topple pushes, protective-fall
costs, recovery rewards) and swaps in jduck. Read that module's docstring for the design
and its 6-run lesson arc.

jduck-specific choices:
  - Teachers / warm start are LOCAL jduck checkpoints (env vars), not microduck's wandb runs:
      JDUCK_VS_WALK   walk policy = warm start + walk anchor (default: the N1 policy,
                      modified firmware K=8, robot-style obs)
      JDUCK_VS_STAND  stand-up teacher (default: newest Mjlab-StandUp-Flat-JDuck run)
    Both must use the same 61-D jduck obs (ADC q, Δq qd, 1-step lag) and K=8 firmware.
  - Robot: JDUCK_GROUNDCONTACT_ROBOT_CFG. jduck has no named servo-housing geoms, so the
    servo_impact cost (and its sensor) are dropped; head/trunk impact use jduck bodies.
  - Heights mapped from microduck's 0.060 (sit) / 0.115 (stand) range to jduck's
    0.108 / 0.133; lying heights measured on jduck (face down 0.025, face up 0.094).
  - Crouch spawns use jduck's fold directions (the sitstand deep squat as the anchor).
"""

import math
import os
from copy import deepcopy
from pathlib import Path

from mjlab.managers import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.robot.jduck_constants import (
    JDUCK_ACTUATOR,
    JDUCK_FW_TAG,
    JDUCK_GROUNDCONTACT_ROBOT_CFG,
    JDUCK_HOME,
    JDUCK_HOME_TRUNK_Z,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_velstand_env_cfg as mv
from mjlab_microduck.tasks.jduck_sitstand_env_cfg import JDUCK_SIT_OVERRIDES, JDUCK_SIT_Z, JDUCK_STAND_Z
from mjlab_microduck.tasks.jduck_standup_env_cfg import JDUCK_PRONE_Z, JDUCK_SUPINE_Z
from mjlab_microduck.tasks.jduck_velocity_env_cfg import (
    JDUCK_STAND,
    JDUCK_STAND_CONTACT_W,
    JDUCK_STAND_JOINT_VEL_W,
    JDUCK_STAND_QUIET_S2,
    JDUCK_STAND_QUIET_W,
    JDUCK_STAND_STILL_W,
    feet_contact_when_standing,
    leg_joint_vel_l2_when_standing,
    quiet_actions_when_standing,
    JDUCK_ANG_EMA,
    JDUCK_FRICTION_SCALE,
    JDUCK_HEAD_BODY_NAMES,
    JDUCK_HEAD_CAPS,
    JDUCK_KP_SCALE,
    JDUCK_LIN_STD,
    JDUCK_ANG_STD,
    JDUCK_LIN_VEL_X,
    JDUCK_LIN_VEL_Y,
    JDUCK_ANG_VEL_Z,
    JDUCK_OBS_ADC,
    JDUCK_OBS_LAG,
    JDUCK_SPEED_SCALE,
    JDUCK_TORQUE_SCALE,
    _clip_ranges,
    joint_pos_adc,
    joint_vel_adc,
    randomize_ubt12hc,
    track_yaw_rate_ema,
)

_RL = Path(__file__).resolve().parents[3]          # third_party/microduck_rl


def _newest(pattern: str) -> str | None:
    c = sorted(_RL.glob(pattern), key=lambda p: p.stat().st_mtime)
    return str(c[-1]) if c else None


JDUCK_VS_WALK = os.environ.get("JDUCK_VS_WALK") or _newest(
    "logs/rsl_rl/jduck_velocity_ubt12hc_k8ff0db0*c0.3/*_N1_k8_calm0.3_caps5/model_3999.pt")
JDUCK_VS_STAND = os.environ.get("JDUCK_VS_STAND") or _newest(
    "logs/rsl_rl/jduck_standup_*k8ff0db0*/*/model_*.pt")

def _map_h(h: float) -> float:
    """A microduck height between its sit (0.060) and stand (0.115) → same fraction for jduck."""
    return JDUCK_SIT_Z + (h - 0.060) / (0.115 - 0.060) * (JDUCK_STAND_Z - JDUCK_SIT_Z)


# jduck deep-crouch anchor for the crouch spawns (fold direction of the sitstand squat,
# a bit deeper): left hip_pitch 0.25, knee −1.40, ankle 1.05 (right mirrored).
JDUCK_CROUCH_ANCHOR = {
    "left_hip_pitch": 0.25, "left_knee": -1.40, "left_ankle": 1.05,
    "right_hip_pitch": -0.25, "right_knee": 1.40, "right_ankle": -1.05,
}


def make_jduck_velstand_env_cfg(play: bool = False, rough: bool = False):
    if JDUCK_HOME != "old":
        import warnings
        warnings.warn("jduck velstand is set up around the old HOME; set JDUCK_HOME=old to train it")
    cfg = mv.make_microduck_velstand_env_cfg(play=play, rough=rough)

    # ── robot ─────────────────────────────────────────────────────────────────
    cfg.scene.entities = {"robot": JDUCK_GROUNDCONTACT_ROBOT_CFG}
    sensors = []
    for s in cfg.scene.sensors:
        if s.name == "servo_ground_contact":
            continue                                 # no named servo-housing geoms on jduck
        if s.name == "head_ground_contact":
            s = deepcopy(s)
            s.primary.pattern = rf"^({'|'.join(JDUCK_HEAD_BODY_NAMES)})$"
        sensors.append(s)
    cfg.scene.sensors = tuple(sensors)
    cfg.rewards.pop("servo_impact", None)
    cfg.curriculum.pop("servo_impact_weight", None)

    # ── heights ───────────────────────────────────────────────────────────────
    hp = cfg.rewards["height_progress"].params
    hp["ceiling"] = JDUCK_STAND_Z
    cfg.rewards["com_upward_velocity"].params["max_height"] = JDUCK_STAND_Z + 0.010
    for name in ("fallen_tax", "recovery_success"):
        p = cfg.rewards[name].params
        for k in ("release_z_above", "up_z"):
            if k in p:
                p[k] = _map_h(p[k]) if p[k] > 0.06 else p[k]
    hb = cfg.rewards["head_pose_bias"].params
    hb["gate_height_low"], hb["gate_height_high"] = _map_h(hb["gate_height_low"]), _map_h(hb["gate_height_high"])
    if "body_pose_tracking" in cfg.rewards:
        cfg.rewards["body_pose_tracking"].params["nominal_height"] = JDUCK_HOME_TRUNK_Z
    ft = cfg.terminations["fallen_too_long"].params
    ft["gate_z_below"] = _map_h(ft["gate_z_below"])
    z = JDUCK_HOME_TRUNK_Z
    cfg.events["reset_base"].params["pose_range"]["z"] = (z + 0.002, z + 0.010)

    # ── fallen / crouch / sit spawns ──────────────────────────────────────────
    pr = cfg.events["random_prone_init"].params
    pr["prone_z_min"], pr["prone_z_max"] = JDUCK_PRONE_Z + 0.01, JDUCK_SUPINE_Z + 0.02
    pr["crouch_params"] = {"z_stand": JDUCK_STAND_Z, "z_deep": JDUCK_SIT_Z - 0.01,
                           "anchor_by_name": JDUCK_CROUCH_ANCHOR}
    if "sit_params" in pr:
        sp = dict(pr["sit_params"])
        sp["sitting_joint_overrides"] = dict(JDUCK_SIT_OVERRIDES)
        sp["sitting_joint_noise_std"] = 0.03
        sp["sitting_tilt_max"] = math.radians(3)
        sp["sitting_z_min"], sp["sitting_z_max"] = JDUCK_SIT_Z, JDUCK_SIT_Z + 0.004
        pr["sit_params"] = sp

    # ── commands / tracking: the jduck walk (as N1 was trained) ───────────────
    tw = cfg.commands["twist"]
    tw.ranges.lin_vel_x, tw.ranges.lin_vel_y, tw.ranges.ang_vel_z = JDUCK_LIN_VEL_X, JDUCK_LIN_VEL_Y, JDUCK_ANG_VEL_Z
    cfg.rewards["track_linear_velocity"].params["std"] = JDUCK_LIN_STD
    if JDUCK_ANG_EMA > 0:
        w = cfg.rewards["track_angular_velocity"].weight
        from mjlab.managers import RewardTermCfg
        cfg.rewards["track_angular_velocity"] = RewardTermCfg(
            func=track_yaw_rate_ema, weight=w,
            params={"std": JDUCK_ANG_STD, "tau": JDUCK_ANG_EMA, "command_name": "twist"})
    else:
        cfg.rewards["track_angular_velocity"].params["std"] = JDUCK_ANG_STD

    # ── standing at zero command: the same terms the N1 walk was trained with ─
    from mjlab.managers import RewardTermCfg
    if JDUCK_STAND == "v2":
        cfg.rewards["stand_feet_contact"] = RewardTermCfg(
            func=feet_contact_when_standing, weight=JDUCK_STAND_CONTACT_W, params={"command_threshold": 0.01})
        cfg.rewards["stand_still"] = RewardTermCfg(
            func=microduck_mdp.stillness_at_zero_command, weight=JDUCK_STAND_STILL_W,
            params={"command_threshold": 0.01, "vel_std": 0.1})
        cfg.rewards["stand_joint_vel"] = RewardTermCfg(
            func=leg_joint_vel_l2_when_standing, weight=JDUCK_STAND_JOINT_VEL_W, params={"command_threshold": 0.01})
    if JDUCK_STAND_QUIET_W > 0:
        cfg.rewards["stand_quiet"] = RewardTermCfg(
            func=quiet_actions_when_standing, weight=JDUCK_STAND_QUIET_W,
            params={"s2": JDUCK_STAND_QUIET_S2, "command_threshold": 0.01})

    # ── head ──────────────────────────────────────────────────────────────────
    if "randomize_head_com" in cfg.events:
        cfg.events["randomize_head_com"].params["asset_cfg"] = SceneEntityCfg("robot", body_names=JDUCK_HEAD_BODY_NAMES)
    head = deepcopy(cfg.commands["head_pose"])
    head.ranges = _clip_ranges(head.ranges, JDUCK_HEAD_CAPS)
    cfg.commands["head_pose"] = head
    if "head_pose_range" in cfg.curriculum:
        st = cfg.curriculum["head_pose_range"].params["range_stages"]
        cfg.curriculum["head_pose_range"].params["range_stages"] = [
            {**s, "ranges": _clip_ranges(s["ranges"], JDUCK_HEAD_CAPS)} for s in st]

    # ── actuator: UBT-12HC replica ────────────────────────────────────────────
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
        a["joint_vel"].func = joint_vel_adc
        a["joint_vel"].noise = None
        for t in ("joint_pos", "joint_vel"):
            a[t].delay_min_lag = JDUCK_OBS_LAG
            a[t].delay_max_lag = JDUCK_OBS_LAG
    return cfg


JduckVelStandRlCfg = deepcopy(mv.MicroduckVelStandRlCfg)
_bc = JduckVelStandRlCfg.algorithm.bc_cfg
if _bc is not None:
    _bc.update({
        "wandb_run_path": None, "checkpoint_path": JDUCK_VS_STAND,
        "anchor_wandb_run_path": None, "anchor_checkpoint_path": JDUCK_VS_WALK,
        "anchor_yaw_remap": None,          # the remap table was calibrated on microduck's walk
    })
JduckVelStandRlCfg.experiment_name = f"jduck_velstand_{JDUCK_ACTUATOR}"
if JDUCK_ACTUATOR == "ubt12hc":
    JduckVelStandRlCfg.experiment_name += f"_{JDUCK_FW_TAG}"
JduckVelStandRlCfg.experiment_name += f"_home{JDUCK_HOME}"
if JDUCK_OBS_ADC:
    JduckVelStandRlCfg.experiment_name += "_adcobs" + (f"_lag{JDUCK_OBS_LAG}" if JDUCK_OBS_LAG else "")
if JDUCK_KP_SCALE:
    JduckVelStandRlCfg.experiment_name += f"_kpdr{JDUCK_KP_SCALE[0]:g}-{JDUCK_KP_SCALE[1]:g}"
JduckVelStandRlCfg.run_name = JduckVelStandRlCfg.experiment_name
JduckVelStandRlCfg.max_iterations = 3000
