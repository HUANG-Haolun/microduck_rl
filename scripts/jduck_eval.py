"""Headless jduck policy check on the CPU, using infer_policy.py's own pieces and the
UBT-12HC firmware-law actuator (same as training).

    uv run --frozen python scripts/jduck_eval.py jduck_walk_fw16.onnx [more.onnx ...]

For each policy:
  - 6 velocity commands × 8 s: displacement, heading drift, and peak |hip_yaw| as a
    fraction of the joint's URDF range (is turning range-limited?)
  - stand still 10 s at zero command: foot lift-offs per second (stepping in
    place?) and drift
"""

import contextlib
import io
import math
import sys

import mujoco
import numpy as np

sys.argv, _argv = sys.argv[:1], sys.argv[1:]
sys.path.insert(0, "scripts")
import infer_policy as ip  # noqa: E402

FW = dict(K=16, FF=15, DB=1, floor=0.0, stall=0.96, w0=4.8)
# `pd:` prefix on an ONNX path = evaluate with the v0 placeholder PD actuator
# (kp=8, kd=0.2, 0.8 N·m) that jduck_walk_v0.onnx was trained with.
CMDS = [(0, 0, 0), (0.2, 0, 0), (-0.2, 0, 0), (0, 0.12, 0), (0, 0, 0.8), (0.15, 0, 0.5)]


class Rig:
    def __init__(self, onnx, pd=False):
        self.pd = pd
        ip.DEFAULT_POSE = ip._onnx_default_pose(onnx)
        self.m = mujoco.MjModel.from_xml_path(ip.JDUCK_XML)
        self.m.opt.timestep = 0.005
        self.d = mujoco.MjData(self.m)
        with contextlib.redirect_stdout(io.StringIO()):
            self.pol = ip.PolicyInference(self.m, self.d, walking_onnx_path=onnx, delay_min_lag=1,
                                          delay_max_lag=3, use_projected_gravity=True, new_cmd_obs=True)
        self.dofs = np.array(self.pol.joint_qvel_indices)
        self.phase = np.random.default_rng(0).random(14)
        if not pd:   # firmware law drives the joints; XML actuators off
            self.m.actuator_gainprm[:, 0] = 0
            self.m.actuator_biasprm[:, :3] = 0
        else:        # v0 placeholder PD = scene_view.xml actuators (kp=8, kd=0.2, 0.8 N·m)
            self.m.actuator_gainprm[:, 0] = 8.0
            self.m.actuator_biasprm[:, 1] = -8.0
            self.m.actuator_biasprm[:, 2] = -0.2
        self.fj = self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")]
        self.yaw_j = [self.m.joint(n).id for n in ("left_hip_yaw", "right_hip_yaw")]
        self.feet = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, n)
                     for n in ("left_foot_collision", "right_foot_collision")]
        self.floor = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    def reset(self):
        m, d, p = self.m, self.d, self.pol
        mujoco.mj_resetData(m, d)
        d.qpos[self.fj + 3:self.fj + 7] = [1, 0, 0, 0]
        d.qpos[self.fj + 2] = 0.5
        for i, q in enumerate(p.joint_qpos_indices):
            d.qpos[q] = p.default_pose[i]
        p.set_position_targets(p.default_pose)
        mujoco.mj_forward(m, d)
        d.qpos[self.fj + 2] += 0.001 - min(d.site("left_foot").xpos[2], d.site("right_foot").xpos[2])
        mujoco.mj_forward(m, d)
        # Fresh episode: clear the action history and the delay buffer, or the
        # previous run's actions leak into the first steps of this one.
        p.last_action[:] = 0
        if getattr(p, "action_buffer", None) is not None:
            for buf in p.action_buffer:
                buf[:] = 0
            p.buffer_index = 0

    def contacts(self):
        c = [False, False]
        for k in range(self.d.ncon):
            g = (self.d.contact[k].geom1, self.d.contact[k].geom2)
            for i, f in enumerate(self.feet):
                if f in g and self.floor in g:
                    c[i] = True
        return c

    def run(self, cmd, T):
        m, d, p = self.m, self.d, self.pol
        self.reset()
        p.set_vel_cmd(*cmd)
        x0 = d.qpos[self.fj:self.fj + 2].copy()
        lifts = 0
        prev = self.contacts()
        yaw_peak = 0.0
        for k in range(int(T / 0.02)):
            with contextlib.redirect_stdout(io.StringIO()):
                p.update_behavior(0.02)
                p.apply_action(p.infer())
            for _ in range(4):
                d.qfrc_applied[:] = 0
                if not self.pd:
                    d.qfrc_applied[self.dofs] = ip.jduck_fw_torque(d.qpos[p.joint_qpos_indices], d.qvel[self.dofs],
                                                                   d.ctrl[:14], self.phase, FW)
                mujoco.mj_step(m, d)
            now = self.contacts()
            lifts += sum(a and not b for a, b in zip(prev, now))
            prev = now
            for j in self.yaw_j:
                # Fraction of the limit used on the side the joint is on (HOME = 0):
                # 1.0 = sitting on the stop. The tight side is only ~20°.
                lo, hi = m.jnt_range[j]
                q = d.qpos[m.jnt_qposadr[j]]
                yaw_peak = max(yaw_peak, q / hi if q > 0 else q / lo)
            if d.xmat[1].reshape(3, 3)[2, 2] < math.cos(math.radians(60)):
                return dict(fell=k * 0.02)
        q = d.qpos[self.fj + 3:self.fj + 7]
        yaw = math.degrees(math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
        dx = d.qpos[self.fj:self.fj + 2] - x0
        return dict(dx=dx, yaw=yaw, lifts_per_s=lifts / T, yaw_peak=yaw_peak)


REPS = 4   # each command is run with every lag 1..3 and different ADC phases


def runs(rig, cmd, T):
    out = []
    for rep in range(REPS):
        rig.pol.current_lag = 1 + rep % 3
        rig.phase = np.random.default_rng(rep).random(14)
        out.append(rig.run(cmd, T))
    return out


def main():
    for arg in _argv:
        pd = arg.startswith("pd:")
        onnx = arg[3:] if pd else arg
        rig = Rig(onnx, pd=pd)
        print(f"\n=== {onnx} [{'PD actuator' if pd else 'K=16 firmware law'}]   (mean ± std over {REPS} runs: lag 1..3, varied ADC phase)")
        for c in CMDS:
            rs = runs(rig, c, 8.0)
            ok = [r for r in rs if "fell" not in r]
            fell = len(rs) - len(ok)
            if not ok:
                print(f"  cmd {str(c):17s} FELL in all {REPS}")
                continue
            dx = np.array([r["dx"] for r in ok])
            yaw = np.array([r["yaw"] for r in ok])
            ideal = np.array(c[:2]) * 8
            err = np.linalg.norm(dx - ideal, axis=1)
            print(f"  cmd {str(c):17s} dx={dx[:, 0].mean():+.2f}±{dx[:, 0].std():.2f} dy={dx[:, 1].mean():+.2f}±{dx[:, 1].std():.2f}"
                  f"  pos err {err.mean():.2f} m  yaw={yaw.mean():+6.0f}±{yaw.std():3.0f}°"
                  f"  (ideal {ideal[0]:+.1f},{ideal[1]:+.1f}, yaw {math.degrees(c[2] * 8):+.0f}°)"
                  f"  hip_yaw {100 * np.mean([r['yaw_peak'] for r in ok]):.0f}%" + (f"  FELL {fell}/{REPS}" if fell else ""))
        rs = [r for r in runs(rig, (0, 0, 0), 10.0) if "fell" not in r]
        if rs:
            print(f"  stand 10 s: {np.mean([r['lifts_per_s'] for r in rs]):.2f} foot lift-offs/s, "
                  f"drift {np.mean([np.linalg.norm(r['dx']) for r in rs]):.2f} m, "
                  f"|yaw| {np.mean([abs(r['yaw']) for r in rs]):.0f}°   (0 = stands still, ~2/s = stepping in place)")


if __name__ == "__main__":
    main()
