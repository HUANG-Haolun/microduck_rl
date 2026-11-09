"""jduck stand-up (get up from lying / squatting): Microduck's StandUp recipe on jduck.

Built for one reason: Mjlab-VelStand-* distils the get-up from a frozen stand-up
"teacher" policy (distill.py, fallen-gated BC). For jduck that teacher has to be a
jduck policy, trained here.

What changes vs microduck_standup_env_cfg.py (same approach as jduck_sitstand_env_cfg.py):
  - Robot: JDUCK_GROUNDCONTACT_ROBOT_CFG (torso, hip-pitch, knee and head meshes collide
    with the floor). Modified firmware (default JDUCK_FW=8,0,0), robot-style joint obs,
    UBT-12HC DR instead of BAM's.
  - Heights measured on jduck (~/jduck_ops/lie_z.py, sit_probe.py): standing 0.133 m (old
    HOME), the "sitting" start = the sitstand deep squat at 0.108 m, lying face down
    0.025 m, face up 0.094 m, on the side 0.073 m. Microduck's 0.060/0.115 → 0.108/0.133.
    Gates and bands that sit between microduck's sit and stand heights are mapped
    linearly into jduck's range.
  - No scripted get-up exists (scripted tuck-and-rise from face down/up fails even with
    unlimited torque, ~/jduck_ops/getup_probe.py) — the policy has to discover one, as
    microduck's did.
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
from mjlab_microduck.tasks import microduck_standup_env_cfg as mu
from mjlab_microduck.tasks.jduck_sitstand_env_cfg import JDUCK_SIT_OVERRIDES, JDUCK_SIT_Z, JDUCK_STAND_Z
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

JDUCK_PRONE_Z = 0.025      # face down at rest
JDUCK_SUPINE_Z = 0.094     # face up at rest


def _map_h(h: float) -> float:
    """Microduck height between its SIT_Z and STAND_Z → the same fraction of jduck's range."""
    f = (h - mu.SIT_Z) / (mu.STAND_Z - mu.SIT_Z)
    return JDUCK_SIT_Z + f * (JDUCK_STAND_Z - JDUCK_SIT_Z)


_SCALE = (JDUCK_STAND_Z - JDUCK_SIT_Z) / (mu.STAND_Z - mu.SIT_Z)

# Every height-like reward parameter, and how it maps. A missing key fails loudly.
_REWARD_HEIGHTS = {
    ("height_stand", "target_height"): "stand",
    ("height_stand", "std"): "scale",
    ("height_stand_sharp", "target_height"): "stand",
    ("height_stand_sharp", "std"): "scale",
    ("height_stand_l1", "target_height"): "stand",
    ("com_upward_velocity", "max_height"): "above_stand",
    ("arrival_damping", "height_low"): "map",
    ("arrival_damping", "height_high"): "map",
    ("upright_sharp", "height_low"): "sit",
    ("upright_sharp", "height_high"): "stand",
    ("standing_composite", "target_height"): "stand",
    ("standing_composite", "height_std"): "scale",
    ("head_pose_bias", "gate_height_low"): "map",
    ("head_pose_bias", "gate_height_high"): "map",
    ("body_pose_tracking", "nominal_height"): "stand",
}


def make_jduck_standup_env_cfg(play: bool = False, rough: bool = False):
    if JDUCK_HOME != "old":
        import warnings
        warnings.warn("jduck standup is set up around the old HOME; set JDUCK_HOME=old to train it")
    cfg = mu.make_microduck_standup_env_cfg(play=play, rough=rough)
    cfg.scene.entities = {"robot": JDUCK_GROUNDCONTACT_ROBOT_CFG}

    for (term, key), how in _REWARD_HEIGHTS.items():
        if term not in cfg.rewards:
            continue
        p = cfg.rewards[term].params
        assert key in p, f"{term}.{key} not found — microduck standup changed?"
        v = p[key]
        p[key] = {"stand": JDUCK_STAND_Z, "sit": JDUCK_SIT_Z, "scale": v * _SCALE,
                  "above_stand": JDUCK_STAND_Z + (v - mu.STAND_Z), "map": _map_h(v)}[how]
    for term in cfg.rewards.values():
        if "target_overrides" in term.params and term.params["target_overrides"]:
            term.params["target_overrides"] = dict(JDUCK_SIT_OVERRIDES)

    # resets: jduck lying / squat / standing heights, squat keyframe
    gs = cfg.events["set_ground_state"].params
    gs["sitting_joint_overrides"] = dict(JDUCK_SIT_OVERRIDES)
    gs["sitting_joint_noise_std"] = 0.03          # the squat is less forgiving (jduck sitstand)
    gs["sitting_tilt_max"] = math.radians(3)
    gs["sitting_z_min"], gs["sitting_z_max"] = JDUCK_SIT_Z, JDUCK_SIT_Z + 0.004
    z = JDUCK_HOME_TRUNK_Z
    gs["standing_z_min"], gs["standing_z_max"] = z + 0.002, z + 0.010
    # prone spawns use one z band for face down and face up: cover both rest heights
    gs["prone_z_min"], gs["prone_z_max"] = JDUCK_PRONE_Z + 0.01, JDUCK_SUPINE_Z + 0.02
    cfg.events["reset_base"].params["pose_range"]["z"] = (z + 0.002, z + 0.010)

    # body-pose command crouch/extension limits are in metres below/above STAND_Z
    if "body_pose" in cfg.commands and hasattr(cfg.commands["body_pose"], "ranges"):
        pass   # ranges are deltas; the scaled z_std above keeps the tracking reward consistent

    # head
    if "randomize_head_com" in cfg.events:
        cfg.events["randomize_head_com"].params["asset_cfg"] = SceneEntityCfg("robot", body_names=JDUCK_HEAD_BODY_NAMES)
    head = deepcopy(cfg.commands["head_pose"])
    head.ranges = _clip_ranges(head.ranges, JDUCK_HEAD_CAPS)
    cfg.commands["head_pose"] = head
    if "head_pose_range" in cfg.curriculum:
        st = cfg.curriculum["head_pose_range"].params["range_stages"]
        cfg.curriculum["head_pose_range"].params["range_stages"] = [
            {**s, "ranges": _clip_ranges(s["ranges"], JDUCK_HEAD_CAPS)} for s in st]

    # actuator: UBT-12HC firmware replica, not BAM
    for name in ("randomize_joint_friction", "randomize_motor_gains"):
        cfg.events.pop(name, None)
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


JduckStandUpRlCfg = deepcopy(mu.MicroduckStandUpRlCfg)
JduckStandUpRlCfg.experiment_name = f"jduck_standup_{JDUCK_ACTUATOR}"
if JDUCK_ACTUATOR == "ubt12hc":
    JduckStandUpRlCfg.experiment_name += f"_{JDUCK_FW_TAG}"
JduckStandUpRlCfg.experiment_name += f"_home{JDUCK_HOME}"
if JDUCK_OBS_ADC:
    JduckStandUpRlCfg.experiment_name += "_adcobs" + (f"_lag{JDUCK_OBS_LAG}" if JDUCK_OBS_LAG else "")
if JDUCK_KP_SCALE:
    JduckStandUpRlCfg.experiment_name += f"_kpdr{JDUCK_KP_SCALE[0]:g}-{JDUCK_KP_SCALE[1]:g}"
JduckStandUpRlCfg.run_name = JduckStandUpRlCfg.experiment_name
JduckStandUpRlCfg.max_iterations = 6000
