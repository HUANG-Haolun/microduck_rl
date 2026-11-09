"""jduck sit ↔ stand: Microduck's SitStand recipe on the jduck robot.

Builds `make_microduck_sitstand_env_cfg` (posture-conditioned rewards, slewed target,
gentle-motion caps, delayed pushes, seated perturbations) and swaps in what is
jduck-specific. The reward design and its lessons are in microduck_sitstand_env_cfg.py.

What changes, and why (all measured in sim with ~/jduck_ops/sit_probe.py, sit_fw.py,
gc_check.py on 2026-11-09):
  - Robot: JDUCK_GROUNDCONTACT_ROBOT_CFG (torso, hip-pitch, knee and head meshes also
    collide with the floor; the walking model only has feet).
  - "Sit" = a deep squat on the feet, NOT butt on the ground. Left leg hip_pitch +0.25,
    knee −1.20, ankle +0.90 (right mirrored): rests at trunk z 0.108 m, tilt 2.9°, and
    with the modified firmware (K=8/16) a plain 2 s target ramp goes HOME → squat →
    HOME 3/3. A butt-down sit (hip −0.30, knee −1.45) is stable too, but even unlimited
    torque cannot rise from it (the COM sits behind the feet) — revisit later.
  - Heights: STAND_Z 0.133 (old HOME), SIT_Z 0.108 — 25 mm apart (microduck: 55 mm), so
    the height bands and speed caps are scaled by 25/55.
  - Firmware: needs the MODIFIED firmware. The stock law (48-PWM floor, deadband 2) falls
    over going HOME → half squat with no policy at all. Default JDUCK_FW=8,0,0 (the same
    servo setting as the N1 walking policy).
  - Joint obs as on the robot (JDUCK_OBS_ADC / JDUCK_OBS_LAG), UBT-12HC torque/speed/KP
    DR, and the BAM-only friction DR replaced, as in the jduck velocity task.

Run:
    JDUCK_FW=8,0,0 JDUCK_TORQUE=0.96 JDUCK_NOLOAD=4.8 \
    uv run --frozen train Mjlab-SitStand-Flat-JDuck --env.scene.num-envs 4096 --agent.max_iterations 6000
"""

import math
import os
from copy import deepcopy

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
from mjlab_microduck.tasks import microduck_sitstand_env_cfg as ms
from mjlab_microduck.tasks.jduck_velocity_env_cfg import (
    JDUCK_FRICTION_SCALE,
    JDUCK_HEAD_BODY_NAMES,
    JDUCK_HEAD_CAPS,
    JDUCK_KP_SCALE,
    JDUCK_OBS_ADC,
    JDUCK_OBS_LAG,
    JDUCK_SPEED_SCALE,
    JDUCK_TORQUE_SCALE,
    _clip_ranges,
    joint_pos_adc,
    joint_vel_adc,
    randomize_ubt12hc,
)


# Sit keyframe, servo indices (14-joint layout; passive_mouth is skipped by the mdp helpers).
JDUCK_SIT_OVERRIDES = {
    1: 0.0,  2: 0.25,  3: -1.20,  4: 0.90,      # left  hip_roll, hip_pitch, knee, ankle
    10: 0.0, 11: -0.25, 12: 1.20, 13: -0.90,     # right (mirrored)
}
JDUCK_STAND_Z = float(os.environ.get("JDUCK_STAND_Z", "0.133"))
JDUCK_SIT_Z = float(os.environ.get("JDUCK_SIT_Z", "0.108"))
_SCALE = (JDUCK_STAND_Z - JDUCK_SIT_Z) / (ms.STAND_Z - ms.SIT_Z)    # 25/55 ≈ 0.45

# Upright gating: microduck uses 0.10 / 0.075 inside its 0.060–0.115 range. Same fractions.
JDUCK_STAND_UPRIGHT_Z = JDUCK_SIT_Z + (ms.STAND_UPRIGHT_Z - ms.SIT_Z) / (ms.STAND_Z - ms.SIT_Z) * (JDUCK_STAND_Z - JDUCK_SIT_Z)
JDUCK_SIT_UPRIGHT_Z = JDUCK_SIT_Z + (ms.SIT_UPRIGHT_Z - ms.SIT_Z) / (ms.STAND_Z - ms.SIT_Z) * (JDUCK_STAND_Z - JDUCK_SIT_Z)


def _replace_heights(params: dict) -> None:
    """Swap microduck's sit pose and sit/stand heights in a term's params."""
    if "sit_z" in params:
        params["sit_z"] = JDUCK_SIT_Z
    if "stand_z" in params:
        params["stand_z"] = JDUCK_STAND_Z
    for key in ("sit_overrides", "sitting_joint_overrides"):
        if key in params:
            params[key] = dict(JDUCK_SIT_OVERRIDES)


# Lengths/speeds in microduck's terms that are tied to its 55 mm sit→stand travel. Scaled by
# the travel ratio; `max_height` (rise bootstrap pays below it) keeps its 10 mm margin above
# STAND_Z. Every (term, key) here must exist, so a renamed parameter fails loudly.
_LENGTH_PARAMS = {
    ("posture_height", "std"): "scale",
    ("posture_height_sharp", "std"): "scale",
    ("posture_stillness", "band_full"): "scale",
    ("posture_stillness", "band_zero"): "scale",
    ("posture_composite", "height_std"): "scale",
    ("rise_bootstrap", "max_height"): "above_stand",
    ("rise_bootstrap", "max_vz"): "scale",
    ("descent_speed", "max_down_vel"): "scale",
    ("rise_speed", "max_up_vel"): "scale",
}


def make_jduck_sitstand_env_cfg(play: bool = False, rough: bool = False):
    # The task package registers every task at import, so this runs in velocity-only processes
    # too: warn instead of failing there. Training this task needs JDUCK_HOME=old.
    if JDUCK_HOME != "old":
        import warnings
        warnings.warn("jduck sitstand is set up around the old HOME; set JDUCK_HOME=old to train it")
    cfg = ms.make_microduck_sitstand_env_cfg(play=play, rough=rough)

    cfg.scene.entities = {"robot": JDUCK_GROUNDCONTACT_ROBOT_CFG}

    # rewards / command / reset: jduck sit pose and heights
    for term in cfg.rewards.values():
        _replace_heights(term.params)
    for (term, key), how in _LENGTH_PARAMS.items():
        p = cfg.rewards[term].params
        assert key in p, f"{term}.{key} not found — microduck sitstand changed?"
        p[key] = p[key] * _SCALE if how == "scale" else JDUCK_STAND_Z + (p[key] - ms.STAND_Z)
    if "upright_while_tall" in cfg.rewards:
        cfg.rewards["upright_while_tall"].params["height_low"] = JDUCK_SIT_UPRIGHT_Z
        cfg.rewards["upright_while_tall"].params["height_high"] = JDUCK_STAND_UPRIGHT_Z
    tw = cfg.commands["twist"]
    tw.sit_z, tw.stand_z = JDUCK_SIT_Z, JDUCK_STAND_Z
    for term in cfg.events.values():
        _replace_heights(term.params)
    z = JDUCK_HOME_TRUNK_Z
    cfg.events["reset_base"].params["pose_range"]["z"] = (z + 0.002, z + 0.010)
    gs = cfg.events["set_ground_state"].params
    # Seated resets: the squat is less forgiving than microduck's butt-down sit. Held open-loop
    # (zero action) for 2 s, microduck's noise (0.10 rad, ±8°) tips 52% over; 0.03 rad, ±3°,
    # z 0.108–0.112 tips 29% (no noise: 0%). The policy balances actively, as it does standing.
    gs["sitting_z_min"], gs["sitting_z_max"] = JDUCK_SIT_Z, JDUCK_SIT_Z + 0.004
    gs["sitting_joint_noise_std"] = 0.03
    gs["sitting_tilt_max"] = math.radians(3)
    gs["standing_z_min"], gs["standing_z_max"] = z + 0.002, z + 0.010

    # head: jduck bodies and joint ranges
    if "randomize_head_com" in cfg.events:
        cfg.events["randomize_head_com"].params["asset_cfg"] = SceneEntityCfg("robot", body_names=JDUCK_HEAD_BODY_NAMES)
    head = deepcopy(cfg.commands["head_pose"])
    head.ranges = _clip_ranges(head.ranges, JDUCK_HEAD_CAPS)
    cfg.commands["head_pose"] = head
    if "head_pose_range" in cfg.curriculum:
        st = cfg.curriculum["head_pose_range"].params["range_stages"]
        cfg.curriculum["head_pose_range"].params["range_stages"] = [
            {**s, "ranges": _clip_ranges(s["ranges"], JDUCK_HEAD_CAPS)} for s in st]

    # actuator: the UBT-12HC firmware replica, not BAM
    for name in ("randomize_joint_friction", "randomize_motor_gains"):
        cfg.events.pop(name, None)            # BAM / XML-PD hooks: no-ops or wrong on jduck
    if JDUCK_ACTUATOR == "ubt12hc":
        cfg.events["randomize_ubt12hc"] = EventTermCfg(
            func=randomize_ubt12hc, mode="reset",
            params={"torque_range": JDUCK_TORQUE_SCALE, "speed_range": JDUCK_SPEED_SCALE,
                    "kp_range": JDUCK_KP_SCALE},
        )
    if JDUCK_FRICTION_SCALE:
        cfg.events["randomize_jduck_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_dof_field_scaled, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",)),
                    "field": "dof_frictionloss", "scale_range": JDUCK_FRICTION_SCALE},
        )

    # joint obs as the robot reads them (jduck_run.py)
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


JduckSitStandRlCfg = deepcopy(ms.MicroduckSitStandRlCfg)
JduckSitStandRlCfg.experiment_name = f"jduck_sitstand_{JDUCK_ACTUATOR}"
if JDUCK_ACTUATOR == "ubt12hc":
    JduckSitStandRlCfg.experiment_name += f"_{JDUCK_FW_TAG}"
JduckSitStandRlCfg.experiment_name += f"_home{JDUCK_HOME}"
if JDUCK_OBS_ADC:
    JduckSitStandRlCfg.experiment_name += "_adcobs" + (f"_lag{JDUCK_OBS_LAG}" if JDUCK_OBS_LAG else "")
if JDUCK_KP_SCALE:
    JduckSitStandRlCfg.experiment_name += f"_kpdr{JDUCK_KP_SCALE[0]:g}-{JDUCK_KP_SCALE[1]:g}"
JduckSitStandRlCfg.run_name = JduckSitStandRlCfg.experiment_name
JduckSitStandRlCfg.max_iterations = 6000
