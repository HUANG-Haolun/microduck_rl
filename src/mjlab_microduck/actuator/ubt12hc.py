"""UBT-12HC servo actuator: a replica of the firmware control law plus a DC motor.

Firmware `send_rev_230400_ult.ino`, `pid_update()` at 1 kHz:

    err = target_adc - adc                       # ADC counts, 1024 / 360° → ~0.35°/count
    if |err| <= 2:  motor off                    # deadband
    pwm = clamp(|err|, 48, 255)                  # Kp/Ki/Kd are declared but NOT used
    drive ±pwm

So it is a P law of exactly 1 PWM/count with a floor of 48 and a ±2-count dead zone.
At errors between 3 and 48 counts (~1–17°) it pushes with a constant 48/255.

Motor: duty × stall torque, limited by a linear torque–speed curve (DC motor with
back-EMF): τ = τ_stall · (duty − ω/ω_noload), clamped to ±τ_stall · duty-range.
τ_stall and ω_noload are the URDF effort / velocity (0.8 N·m, 3.0 rad/s), which
the user's friend says are right. The PWM→torque mapping is assumed linear; bench
identification should replace these numbers.

Sim vs firmware rate: MuJoCo steps at 5 ms, the firmware at 1 ms. The torque is
computed once per physics step from the current (quantized) position, so within
one step the law acts like a relay with a deadband. The back-EMF term supplies the
damping the real motor has, which keeps a 5 ms relay from chattering on its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg

if TYPE_CHECKING:
    from mjlab.entity import Entity

RAD_PER_COUNT = 2.0 * math.pi / 1024.0


@dataclass(kw_only=True)
class Ubt12hcActuatorCfg(IdealPdActuatorCfg):
    """Firmware-replica servo. `stiffness`/`damping` are unused (kept for the base class)."""

    stall_torque: float = 0.8          # N·m at duty 1.0 (URDF effort)
    no_load_speed: float = 3.0         # rad/s at duty 1.0 (URDF velocity)
    pwm_min: float = 48.0              # firmware MIN_OUTPUT_PWM
    pwm_max: float = 255.0             # firmware MAX_output_pwm
    pwm_per_count: float = 1.0         # firmware: pwm = |err| (Kp unused)
    deadband_counts: float = 2.0       # firmware POSITION_DEADBAND
    rad_per_count: float = RAD_PER_COUNT
    quantize: bool = True              # the firmware sees integer ADC counts

    def build(self, entity: Entity, target_ids: list[int], target_names: list[str]) -> "Ubt12hcActuator":
        return Ubt12hcActuator(self, entity, target_ids, target_names)


class Ubt12hcActuator(IdealPdActuator[Ubt12hcActuatorCfg]):
    """Per-env `torque_scale` (stall torque) and `speed_scale` can be randomized."""

    def initialize(self, mj_model, model, data, device: str) -> None:
        super().initialize(mj_model, model, data, device)
        shape = (data.nworld, len(self._target_names))
        self.torque_scale = torch.ones(shape, dtype=torch.float, device=device)
        self.speed_scale = torch.ones(shape, dtype=torch.float, device=device)
        # Per-env, per-joint encoder phase in [0, 1) counts: where the integer ADC
        # boundaries fall relative to the sim's continuous angle.
        self.adc_phase = torch.rand(shape, dtype=torch.float, device=device)

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        c = self.cfg
        k = c.rad_per_count
        if c.quantize:
            pos_counts = torch.floor(cmd.pos / k + self.adc_phase)
            tgt_counts = torch.round(cmd.position_target / k + self.adc_phase)
            err = tgt_counts - pos_counts
        else:
            err = (cmd.position_target - cmd.pos) / k

        mag = torch.clamp(err.abs() * c.pwm_per_count, c.pwm_min, c.pwm_max)
        pwm = torch.where(err.abs() <= c.deadband_counts, torch.zeros_like(mag), mag)
        duty = torch.sign(err) * pwm / 255.0

        tau_stall = c.stall_torque * self.torque_scale
        w0 = c.no_load_speed * self.speed_scale
        torque = tau_stall * (duty - cmd.vel / w0)
        # Motor off in the deadband: the H-bridge output is off, no back-EMF braking
        # (the driver's sleep pin stays HIGH, so this is coast, not brake).
        torque = torch.where(pwm == 0, torch.zeros_like(torque), torque)
        return torch.clamp(torque, -tau_stall, tau_stall)

    def set_scales(self, env_ids, torque_scale=None, speed_scale=None) -> None:
        if torque_scale is not None:
            self.torque_scale[env_ids] = torque_scale
        if speed_scale is not None:
            self.speed_scale[env_ids] = speed_scale

    def reset_encoder_phase(self, env_ids) -> None:
        n = self.adc_phase[env_ids].shape
        self.adc_phase[env_ids] = torch.rand(n, device=self.adc_phase.device)
