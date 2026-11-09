"""jduck cfg invariants: model resolves, 14 servos in microduck order, mouth passive."""

import mujoco
import pytest

from mjlab_microduck.robot.jduck_constants import JDUCK_WALK_XML, get_jduck_walk_spec

pytestmark = pytest.mark.skipif(
    not JDUCK_WALK_XML.exists(), reason="jduck MJCF not built (jduck_mjcf/build_mjcf.py)"
)

SERVOS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]


def _model():
    return get_jduck_walk_spec().compile()


def test_joint_layout_matches_microduck():
    m = _model()
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    assert names[0] == "trunk_base_freejoint"
    servo_names = [n for n in names[1:] if not n.startswith("passive_")]
    assert servo_names == SERVOS
    assert "passive_mouth" in names


def test_names_the_velocity_recipe_hardcodes():
    m = _model()
    for body in ("trunk_base", "neck_pitch_link", "head_pitch_link", "head_yaw_link", "head_roll_link"):
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body) >= 0, body
    for site in ("left_foot", "right_foot", "imu"):
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site) >= 0, site
    for geom in ("left_foot_collision", "right_foot_collision"):
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom) >= 0, geom


def test_no_xml_actuators_and_measured_mass():
    m = _model()
    assert m.nu == 0  # mjlab's actuator cfg adds them; duplicates fail to compile
    assert m.body_subtreemass[1] == pytest.approx(1.366, abs=0.01)


def test_left_right_symmetric_at_zero_pose():
    m = _model()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for b in ("hip_yaw_link", "hip_pitch_link", "knee_link", "ankle_link"):
        left, right = d.body(f"left_{b}").xpos, d.body(f"right_{b}").xpos
        assert left[0] == pytest.approx(right[0], abs=1e-4)
        assert left[1] == pytest.approx(-right[1], abs=1e-4)
        assert left[2] == pytest.approx(right[2], abs=1e-4)


def test_jduck_stand_action_rate_penalty():
    """JDUCK_STAND_ACT_RATE adds a negative-weight zero-command action-rate term.

    Runs in a subprocess: this file imports jduck_constants first, so the task
    package's import-time registration is already half done in this process.
    """
    import os
    import subprocess
    import sys

    code = (
        "import mjlab_microduck.tasks.jduck_velocity_env_cfg as m\n"
        "c = m.make_jduck_velocity_env_cfg()\n"
        "t = c.rewards.get('stand_action_rate')\n"
        "print(None if t is None else t.weight, m.JduckRlCfg.experiment_name)\n"
    )
    def run(w):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("JDUCK_STAND_ACT_RATE", "JDUCK_SAR_"))}
        if w is not None:
            env["JDUCK_STAND_ACT_RATE"] = w
            env["JDUCK_SAR_START"] = "0"     # weight from iteration 0 (the staged default starts at 0)
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        return out.stdout.strip().splitlines()[-1].split()

    weight, name = run(None)
    assert weight == "None" and "_sar" not in name          # default off: old runs rebuild unchanged
    weight, name = run("20")
    assert float(weight) == -20.0 and name.endswith("_sar20")


def test_jduck_staged_action_rate_and_yaw_ema():
    """Staged stand_action_rate (0 → ¼ → ½ → full) and the EMA yaw-rate term."""
    import os
    import subprocess
    import sys

    code = (
        "import mjlab_microduck.tasks.jduck_velocity_env_cfg as m\n"
        "c = m.make_jduck_velocity_env_cfg()\n"
        "print(c.rewards['stand_action_rate'].weight)\n"
        "print([s['weight'] for s in c.curriculum['stand_action_rate_weight'].params['weight_stages']])\n"
        "print(c.rewards['track_angular_velocity'].func.__name__, c.rewards['track_angular_velocity'].weight)\n"
        "print(m.JduckRlCfg.experiment_name)\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("JDUCK_")}
    env.update(JDUCK_STAND_ACT_RATE="10", JDUCK_ANG_EMA="0.5")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    w0, stages, ang, name = out.stdout.strip().splitlines()[-4:]
    assert float(w0) == 0.0
    assert stages == "[0.0, -2.5, -5.0, -10.0]"
    assert ang == "track_yaw_rate_ema 2.0"
    assert name.endswith("_sar10from1500_angema0.5")


def test_groundcontact_model_matches_walk_model():
    """robot_groundcontact.xml = robot_walk.xml + floor-colliding body meshes; same joints and mass."""
    from mjlab_microduck.robot.jduck_constants import get_jduck_groundcontact_spec

    w = _model()
    g = get_jduck_groundcontact_spec().compile()
    names = lambda m, t, n: [mujoco.mj_id2name(m, t, i) for i in range(n)]
    assert names(w, mujoco.mjtObj.mjOBJ_JOINT, w.njnt) == names(g, mujoco.mjtObj.mjOBJ_JOINT, g.njnt)
    assert abs(w.body_subtreemass[1] - g.body_subtreemass[1]) < 1e-9
    floor_geoms = {mujoco.mj_id2name(g, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(g.ngeom) if g.geom_contype[i] & 1}
    assert "trunk_base_ground_collision" in floor_geoms
    assert {"left_foot_collision", "right_foot_collision"} <= floor_geoms
    walk_floor = {mujoco.mj_id2name(w, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(w.ngeom) if w.geom_contype[i] & 1}
    assert walk_floor == {"left_foot_collision", "right_foot_collision"}   # walking model unchanged
