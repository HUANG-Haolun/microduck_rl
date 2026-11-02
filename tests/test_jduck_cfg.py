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
    assert names[0] == "root"
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
