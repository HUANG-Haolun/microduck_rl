"""PPO + CAPS-style action smoothness (Mysore et al., "Regularizing Action Policies for
Smooth Control with Reinforcement Learning", ICRA 2021).

Why: jduck policies stand still with clean observations but jitter (2–10°/step) once
q is ADC-quantized and qd is a finite difference, both in their own training env
(JDUCK_OBS_ADC=1) and through jduck_run.py. Reward shaping (action-rate penalty,
quiet-stand bonus) did not make the policy insensitive to that sensor noise. CAPS
regularizes the policy function itself:

  spatial   L_S = ‖π(o) − π(o + ε)‖²,     ε ~ N(0, σ_sensor)   (insensitive to sensor noise)
  temporal  L_T = ‖π(o_t) − π(o_{t+1})‖²   (consecutive steps of the same episode)

Both act on the deterministic action mean. After the normal PPO update, a few extra
Adam steps minimize λ_S·L_S + λ_T·L_T on the rollout just used. Samples with a
zero twist command get weight `stand_boost` (that is where the jitter shows; walking
needs fast action changes, so it only gets the base weight).

Only the actor is trained here; the exported ONNX is unchanged in structure.
"""

from __future__ import annotations

import torch
from rsl_rl.algorithms import PPO

# Actor obs layout (61D): ang_vel 0:3 | gravity 3:6 | q 6:20 | qd 20:34 | last_action 34:48 | cmd 48:61
_K = 2 * torch.pi / 1024          # rad per ADC count
_SIGMA = (
    [0.05] * 3                    # gyro, rad/s
    + [0.02] * 3                  # projected gravity
    + [_K] * 14                   # q: ±1 ADC count
    + [_K / 0.02] * 14            # qd: one count over one 20 ms step ≈ 0.31 rad/s
    + [0.0] * 14                  # last action: the policy's own output, exact
    + [0.0] * 13                  # command: exact
)


class PPOCaps(PPO):
    def __init__(self, *args, caps_spatial: float = 0.0, caps_temporal: float = 0.0,
                 caps_stand_boost: float = 4.0, caps_walk_weight: float = 1.0, caps_calm: float = 0.0,
                 caps_steps: int = 4, caps_batch: int = 16384,
                 caps_sigma_scale: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.caps_spatial = caps_spatial
        self.caps_temporal = caps_temporal
        self.caps_stand_boost = caps_stand_boost
        self.caps_walk_weight = caps_walk_weight
        self.caps_calm = caps_calm
        self.caps_steps = caps_steps
        self.caps_batch = caps_batch
        self.caps_sigma = torch.tensor(_SIGMA, device=self.device) * caps_sigma_scale

    def update(self) -> dict[str, float]:
        use = self.caps_spatial > 0 or self.caps_temporal > 0
        if use:
            obs = self.storage.observations["actor"].clone()          # (T, N, 61)
            dones = self.storage.dones.clone().squeeze(-1).bool()       # (T, N)
        loss_dict = super().update()
        if use:
            loss_dict.update(self._caps_update(obs, dones))
        return loss_dict

    def _weights(self, o: torch.Tensor) -> torch.Tensor:
        cmd = o[..., 48:51]
        standing = (cmd.abs().sum(-1) < 0.01).float()
        if self.caps_calm > 0:
            # only calm standing: after a push the body rotates and the policy must react
            # fast. Gyro (obs 0:3) above caps_calm rad/s → no smoothing on that sample.
            standing = standing * (o[..., 0:3].norm(dim=-1) < self.caps_calm).float()
        return self.caps_walk_weight * (1.0 - standing) + self.caps_stand_boost * standing

    def _caps_update(self, obs: torch.Tensor, dones: torch.Tensor) -> dict[str, float]:
        from tensordict import TensorDict

        T, N, D = obs.shape
        if D != self.caps_sigma.numel():
            return {}
        flat = obs.reshape(T * N, D)
        # temporal pairs (o_t, o_{t+1}) inside one episode
        nxt_ok = ~dones[:-1].reshape(-1)
        cur = obs[:-1].reshape(-1, D)[nxt_ok]
        nxt = obs[1:].reshape(-1, D)[nxt_ok]

        def act(x):
            return self.actor(TensorDict({"actor": x}, batch_size=[x.shape[0]]))

        ls_sum = lt_sum = 0.0
        for _ in range(self.caps_steps):
            loss = 0.0
            if self.caps_spatial > 0:
                idx = torch.randint(0, flat.shape[0], (min(self.caps_batch, flat.shape[0]),), device=flat.device)
                o = flat[idx]
                o_n = o + torch.randn_like(o) * self.caps_sigma
                ls = (self._weights(o) * (act(o) - act(o_n)).pow(2).sum(-1)).mean()
                loss = loss + self.caps_spatial * ls
                ls_sum += ls.item()
            if self.caps_temporal > 0 and cur.shape[0] > 0:
                idx = torch.randint(0, cur.shape[0], (min(self.caps_batch, cur.shape[0]),), device=cur.device)
                a, b = cur[idx], nxt[idx]
                lt = (self._weights(a) * (act(a) - act(b)).pow(2).sum(-1)).mean()
                loss = loss + self.caps_temporal * lt
                lt_sum += lt.item()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.optimizer.step()
        n = max(1, self.caps_steps)
        out = {}
        if self.caps_spatial > 0:
            out["caps_spatial"] = ls_sum / n
        if self.caps_temporal > 0:
            out["caps_temporal"] = lt_sum / n
        return out
