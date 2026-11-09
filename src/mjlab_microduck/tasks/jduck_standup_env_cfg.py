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


# ── Height rewards gated on being upright ────────────────────────────────────
# jduck's head is long (trunk origin → head top 118 mm, head+neck 36 % of the mass): tipped
# 70° forward onto its head (a "head tripod"), the trunk sits at 0.13 m — the STANDING
# height. jsu1 (5000 it) parked there from every fallen start: 0 % upright from face down /
# up / side, while height_stand paid 0.78 of 1. On microduck a head-down pose is far below
# standing height, so its height terms never had to care about tilt. Multiplying them by an
# upright gate (1 below 25° tilt → 0 at 50°) makes the tripod worth nothing.
def _upright_gate(env, full_deg: float = 25.0, zero_deg: float = 50.0):
    import torch

    g = env.scene["robot"].data.projected_gravity_b[:, 2]
    tilt = torch.rad2deg(torch.arccos(torch.clamp(-g, -1.0, 1.0)))
    return torch.clamp((zero_deg - tilt) / (zero_deg - full_deg), 0.0, 1.0)


def height_target_gaussian_upright(env, target_height: float, std: float = 0.02, asset_cfg=None, **kw):
    return microduck_mdp.height_target_gaussian(env, target_height=target_height, std=std) * _upright_gate(env)


def height_l1_penalty_upright(env, target_height: float, asset_cfg=None, **kw):
    # a cost: keep it at its full (negative) value when tilted, so lying on the head
    # is charged the full distance to standing instead of being treated as "arrived"
    import torch

    asset = env.scene["robot"]
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    gate = _upright_gate(env)
    return -(gate * torch.abs(z - target_height) + (1 - gate) * target_height)


JDUCK_GATE_HEIGHT = os.environ.get("JDUCK_GATE_HEIGHT", "1") == "1"
_make_orig = make_jduck_standup_env_cfg


def make_jduck_standup_env_cfg(play: bool = False, rough: bool = False):   # noqa: F811
    cfg = _make_orig(play=play, rough=rough)
    if JDUCK_GATE_HEIGHT:
        for name in ("height_stand", "height_stand_sharp"):
            if name in cfg.rewards:
                cfg.rewards[name].func = height_target_gaussian_upright
        if "height_stand_l1" in cfg.rewards:
            cfg.rewards["height_stand_l1"].func = height_l1_penalty_upright
    return cfg


if JDUCK_GATE_HEIGHT:
    JduckStandUpRlCfg.experiment_name += "_hgate"
    JduckStandUpRlCfg.run_name = JduckStandUpRlCfg.experiment_name


# ── Head-support cost ────────────────────────────────────────────────────────
# jsu3 (height rewards gated on upright) stopped parking at standing height on its head, but
# from face down it still settles head-on-floor in a crouch (trunk 0.087 m, tilt 25°): the
# head is its third leg. Charge head–floor contact force while the robot should be getting
# up (always, in this task: every episode's goal is standing). Free below 3 N so a brief
# touch while rolling is cheap; ~0.1 per N above that (head+neck weigh ~4.8 N, so leaning
# its full weight on the head costs ~0.2/step ≈ the upright terms it would gain).
from mjlab.sensor import ContactMatch, ContactSensorCfg  # noqa: E402


def head_floor_force(env, sensor_name: str = "head_ground_contact", threshold: float = 3.0):
    import torch

    f = env.scene.sensors[sensor_name].data.force
    f = torch.nan_to_num(f.norm(dim=-1), nan=0.0)
    if f.dim() > 1:
        f = f.sum(dim=-1)
    return torch.clamp(f - threshold, min=0.0)


JDUCK_HEAD_COST = float(os.environ.get("JDUCK_HEAD_COST", "0"))
_make_gated = make_jduck_standup_env_cfg


def make_jduck_standup_env_cfg(play: bool = False, rough: bool = False):   # noqa: F811
    cfg = _make_gated(play=play, rough=rough)
    if JDUCK_HEAD_COST > 0:
        from mjlab.managers import RewardTermCfg

        head = ContactSensorCfg(
            name="head_ground_contact",
            primary=ContactMatch(mode="body", pattern=rf"^({'|'.join(JDUCK_HEAD_BODY_NAMES)})$", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("force",), reduce="netforce", num_slots=1,
        )
        cfg.scene.sensors = tuple(cfg.scene.sensors) + (head,)
        cfg.rewards["head_support"] = RewardTermCfg(func=head_floor_force, weight=-JDUCK_HEAD_COST,
                                                    params={"threshold": 3.0})
    return cfg


if JDUCK_HEAD_COST > 0:
    JduckStandUpRlCfg.experiment_name += f"_head{JDUCK_HEAD_COST:g}"
    JduckStandUpRlCfg.run_name = JduckStandUpRlCfg.experiment_name


# ── Reverse-curriculum spawns from a demonstrated get-up ─────────────────────
# jsu1–jsu6 never discovered a get-up from face down, yet one exists under the servo law:
# ~/jduck_ops/getup_search.py (CEM over 3 symmetric keyframes, firmware K=8, 0.96 N·m) finds
# one, and replays survive K 8–10 / τ 0.96–1.05 / any yaw (getup_record.py). Those replays
# are saved as full (qpos, qvel) every 20 ms. With probability JDUCK_REPLAY_PROB a reset
# overwrites the spawn with a random state along that path (random yaw, env origin), so the
# frontier of the get-up gets on-policy data (AGENTS.md: reverse-curriculum spawns).
JDUCK_REPLAY_PROB = float(os.environ.get("JDUCK_REPLAY_PROB", "0"))
JDUCK_REPLAY_FILE = os.environ.get("JDUCK_REPLAY_FILE",
                                   os.path.expanduser("~/jduck_ops/getup_states_down.npz"))
_REPLAY_CACHE = {}


def spawn_from_replay(env, env_ids, prob: float, path: str):
    import numpy as np
    import torch

    if env_ids is None or len(env_ids) == 0 or prob <= 0:
        return
    key = (path, str(env.device))
    if key not in _REPLAY_CACHE:
        s = np.load(path)["s"]
        _REPLAY_CACHE[key] = torch.tensor(s, dtype=torch.float32, device=env.device)
    S = _REPLAY_CACHE[key]
    nq = env.sim.data.qpos.shape[1]
    assert S.shape[1] == nq + env.sim.data.qvel.shape[1], "replay states do not match this model"
    ids = env_ids[torch.rand(len(env_ids), device=env.device) < prob]
    if len(ids) == 0:
        return
    st = S[torch.randint(0, S.shape[0], (len(ids),), device=env.device)]
    qpos, qvel = st[:, :nq].clone(), st[:, nq:].clone()
    yaw = (torch.rand(len(ids), device=env.device) * 2 - 1) * math.pi
    c, s_ = torch.cos(yaw), torch.sin(yaw)
    # rotate root position/orientation/linear+angular velocity about world z
    x, y = qpos[:, 0], qpos[:, 1]
    qpos[:, 0], qpos[:, 1] = c * x - s_ * y, s_ * x + c * y
    hw, hz = torch.cos(yaw / 2), torch.sin(yaw / 2)
    w, qx, qy, qz = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
    qpos[:, 3:7] = torch.stack([hw * w - hz * qz, hw * qx - hz * qy, hw * qy + hz * qx, hw * qz + hz * w], 1)
    for a in (0, 3):
        vx, vy = qvel[:, a], qvel[:, a + 1]
        qvel[:, a], qvel[:, a + 1] = c * vx - s_ * vy, s_ * vx + c * vy
    origins = env.scene.env_origins[ids]
    qpos[:, 0] = qpos[:, 0] - qpos[:, 0] + origins[:, 0]     # recorded x/y drift dropped
    qpos[:, 1] = origins[:, 1]
    qpos[:, 2] = qpos[:, 2] + origins[:, 2]
    env.sim.data.qpos[ids.long()] = qpos
    env.sim.data.qvel[ids.long()] = qvel


_make_head = make_jduck_standup_env_cfg


def make_jduck_standup_env_cfg(play: bool = False, rough: bool = False):   # noqa: F811
    cfg = _make_head(play=play, rough=rough)
    if JDUCK_REPLAY_PROB > 0:
        cfg.events["spawn_from_replay"] = EventTermCfg(
            func=spawn_from_replay, mode="reset",
            params={"prob": JDUCK_REPLAY_PROB, "path": JDUCK_REPLAY_FILE})
    return cfg


if JDUCK_REPLAY_PROB > 0:
    JduckStandUpRlCfg.experiment_name += f"_replay{JDUCK_REPLAY_PROB:g}"
    JduckStandUpRlCfg.run_name = JduckStandUpRlCfg.experiment_name


# ── Imitation of the demonstrated get-up (DeepMimic-style) ───────────────────
# jsu8 (replay spawns alone) holds the already-standing replay states but does not carry a
# mid-get-up state through: from a spawn partway up it sinks back. An episode spawned at
# frame i of a replay is also paid for tracking frame i+t of the same replay (joint angles
# and trunk height), until that replay ends. Other episodes get 0 from this term.
# Recorded files carry L = frames per replay (20 ms each).
JDUCK_IMIT_W = float(os.environ.get("JDUCK_IMIT_W", "0"))


def _replay_index(env):
    import torch

    if not hasattr(env, "_jduck_replay_idx"):
        env._jduck_replay_idx = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    return env._jduck_replay_idx


def spawn_from_replay_tracked(env, env_ids, prob: float, path: str):
    import numpy as np
    import torch

    if env_ids is None or len(env_ids) == 0:
        return
    idx = _replay_index(env)
    idx[env_ids.long()] = -1
    key = (path, str(env.device))
    if key not in _REPLAY_CACHE:
        f = np.load(path)
        _REPLAY_CACHE[key] = torch.tensor(f["s"], dtype=torch.float32, device=env.device)
        _REPLAY_CACHE[key + ("L",)] = int(f["L"]) if "L" in f else 0
    S, L = _REPLAY_CACHE[key], _REPLAY_CACHE[key + ("L",)]
    sel = env_ids[torch.rand(len(env_ids), device=env.device) < prob]
    if len(sel) == 0:
        return
    pick = torch.randint(0, S.shape[0], (len(sel),), device=env.device)
    # reuse the plain spawn for the state write, with a fixed pick
    gen = torch.Generator(device=env.device)
    st_ids = sel
    _spawn_with_pick(env, st_ids, S, pick)
    idx[st_ids.long()] = pick
    env._jduck_replay_L = L
    env._jduck_replay_S = S


def _spawn_with_pick(env, ids, S, pick):
    import torch

    nq = env.sim.data.qpos.shape[1]
    st = S[pick]
    qpos, qvel = st[:, :nq].clone(), st[:, nq:].clone()
    yaw = (torch.rand(len(ids), device=env.device) * 2 - 1) * math.pi
    c, s_ = torch.cos(yaw), torch.sin(yaw)
    hw, hz = torch.cos(yaw / 2), torch.sin(yaw / 2)
    w, qx, qy, qz = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
    qpos[:, 3:7] = torch.stack([hw * w - hz * qz, hw * qx - hz * qy, hw * qy + hz * qx, hw * qz + hz * w], 1)
    for a in (0, 3):
        vx, vy = qvel[:, a].clone(), qvel[:, a + 1].clone()
        qvel[:, a], qvel[:, a + 1] = c * vx - s_ * vy, s_ * vx + c * vy
    o = env.scene.env_origins[ids]
    qpos[:, 0], qpos[:, 1], qpos[:, 2] = o[:, 0], o[:, 1], qpos[:, 2] + o[:, 2]
    env.sim.data.qpos[ids.long()] = qpos
    env.sim.data.qvel[ids.long()] = qvel


def imitate_replay(env, joint_std: float = 0.3, z_std: float = 0.02):
    import torch

    idx = _replay_index(env)
    out = torch.zeros(env.num_envs, device=env.device)
    L = getattr(env, "_jduck_replay_L", 0)
    if L <= 0:
        return out
    S = env._jduck_replay_S
    t = env.episode_length_buf.long()
    f = idx + t
    ok = (idx >= 0) & (f // L == idx // L)            # still inside the same replay
    if not ok.any():
        return out
    ref = S[f[ok]]
    q = env.sim.data.qpos[ok]
    z = q[:, 2] - env.scene.env_origins[ok, 2]
    jerr = ((q[:, 7:] - ref[:, 7:S.shape[1] - env.sim.data.qvel.shape[1]]) ** 2).mean(-1)
    out[ok] = torch.exp(-jerr / joint_std ** 2) * torch.exp(-((z - ref[:, 2]) ** 2) / z_std ** 2)
    return out


_make_replay = make_jduck_standup_env_cfg


def make_jduck_standup_env_cfg(play: bool = False, rough: bool = False):   # noqa: F811
    cfg = _make_replay(play=play, rough=rough)
    if JDUCK_IMIT_W > 0 and JDUCK_REPLAY_PROB > 0:
        from mjlab.managers import RewardTermCfg

        cfg.events["spawn_from_replay"].func = spawn_from_replay_tracked
        cfg.rewards["imitate_replay"] = RewardTermCfg(func=imitate_replay, weight=JDUCK_IMIT_W)
    return cfg


if JDUCK_IMIT_W > 0:
    JduckStandUpRlCfg.experiment_name += f"_imit{JDUCK_IMIT_W:g}"
if os.environ.get("JDUCK_TORQUE_DR"):
    JduckStandUpRlCfg.experiment_name += f"_tdr{JDUCK_TORQUE_SCALE[0]:g}-{JDUCK_TORQUE_SCALE[1]:g}"
    JduckStandUpRlCfg.run_name = JduckStandUpRlCfg.experiment_name
