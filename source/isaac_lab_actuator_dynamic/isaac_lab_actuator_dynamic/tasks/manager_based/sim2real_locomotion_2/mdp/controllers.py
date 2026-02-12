from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn


@dataclass
class MotorControlCfg:
    joint_name: str
    joint_index: int

    # friction model
    b: float                 # viscous damping [Nm/(rad/s)]
    tau_c: float             # Coulomb magnitude [Nm]

    # inertia / limits
    I_total: float           # effective inertia 
    tau_s_max: float         # soft-limit for STA torque [Nm]

    # STA gains (preferred)
    k1: Optional[float] = None  # [Nm / sqrt(rad/s)]
    k2: Optional[float] = None  # [Nm / s]

    # shaping (used only if k1/k2 missing)
    lambda_: Optional[float] = None  # [rad/s]
    fn: Optional[float] = None       # [Hz], lambda = 2*pi*fn

STA_CONTROLLERS_CFG: Sequence[MotorControlCfg] = [
    MotorControlCfg(
        joint_name="L_hip_joint",
        joint_index=0,
        b=0.179873,
        tau_c=0.0,
        I_total=0.15936282 + 0.075,
        tau_s_max=70.0,
        k1=1.1 * (0.15936282 + 0.075) * math.sqrt(3000.0), # k1_scale * I_total * lam (=2*pi*fn)
        k2=0.2 * (0.15936282 + 0.075) * 3000.0, # k2_scale * I_total * (lam * lam)
        # k1=15.0, # Nm
        # k2=3.0, # Nm/s
        fn=2.88
    ),
    MotorControlCfg(
        joint_name="R_hip_joint",
        joint_index=1,
        b=0.179873,
        tau_c=0.0,
        I_total=0.15936282 + 0.075,
        tau_s_max=70.0,
        k1=1.1 * (0.15936282 + 0.075) * math.sqrt(3000.0), # k1_scale * I_total * lam (=2*pi*fn)
        k2=0.2 * (0.15936282 + 0.075) * 3000.0, # k2_scale * I_total * (lam * lam)
        fn=2.88
    ),
    MotorControlCfg(
        joint_name="L_hip2_joint",
        joint_index=2,
        b=0.593230,
        tau_c=0.0,
        I_total=0.46929292 + 0.125722,
        tau_s_max=70.0,
        k1=1.1 * (0.46929292 + 0.125722) * math.sqrt(466.0), # k1_scale * I_total * lam (=2*pi*fn)
        k2=0.2 * (0.46929292 + 0.125722) * 466.0, # k2_scale * I_total * (lam * lam)
        fn=2.88
    ),
    MotorControlCfg(
        joint_name="R_hip2_joint",
        joint_index=3,
        b=0.593230,
        tau_c=0.0,
        I_total=0.46929292 + 0.125722,
        tau_s_max=70.0,
        k1=1.1 * (0.46929292 + 0.125722) * math.sqrt(466.0), # k1_scale * I_total * lam (=2*pi*fn)
        k2=0.2 * (0.46929292 + 0.125722) * 466.0, # k2_scale * I_total * (lam * lam)
        fn=2.88
    ),
    MotorControlCfg(
        joint_name="L_thigh_joint",
        joint_index=4,
        b=0.686212,
        tau_c=0.0,
        I_total=0.53743194 + 0.17,
        tau_s_max=70.0,
        k1=1.1 * (0.53743194 + 0.17) * math.sqrt(330.0), # k1_scale * I_total * lam (=2*pi*fn)
        k2=0.2 * (0.53743194 + 0.17) * 330.0, # k2_scale * I_total * (lam * lam)
        fn=2.88
    ),
    MotorControlCfg(
        joint_name="R_thigh_joint",
        joint_index=5,
        b=0.686212,
        tau_c=0.0,
        I_total=0.53743194 + 0.17,
        tau_s_max=70.0,
        k1=1.1 * (0.53743194 + 0.17) * math.sqrt(330.0), # k1_scale * I_total * lam (=2*pi*fn)
        k2=0.2 * (0.53743194 + 0.17) * 330.0, # k2_scale * I_total * (lam * lam)
        fn=2.88
    ),
    MotorControlCfg(
        joint_name="L_calf_joint",
        joint_index=6,
        b=3.834728,
        tau_c=0.0,
        I_total=0.11884509 + 0.142563,
        tau_s_max=70.0 * 1.5,
        k1=1.1 * (0.11884509 + 0.142563) * math.sqrt(2412.0), # k1_scale * I_total * lam (=2*pi*fn)
        k2=0.2 * (0.11884509 + 0.142563) * 2412.0, # k2_scale * I_total * (lam * lam)
        fn=2.88
    ),
    MotorControlCfg(
        joint_name="R_calf_joint",
        joint_index=7,
        b=3.834728,
        tau_c=0.0,
        I_total=0.11884509 + 0.142563,
        tau_s_max=70.0 * 1.5,
        k1=1.1 * (0.11884509 + 0.142563) * math.sqrt(2412.0), # k1_scale * I_total * lam (=2*pi*fn)
        k2=0.2 * (0.11884509 + 0.142563) * 2412.0, # k2_scale * I_total * (lam * lam)
        fn=2.88
    ),
    MotorControlCfg(
        joint_name="L_toe_joint",
        joint_index=8,
        b=0.0,
        tau_c=0.0,
        I_total=0.00202688 + 0.045970,
        tau_s_max=25.0,
        k1=1.1 * (0.00202688 + 0.045970) * math.sqrt(71530.0), # k1_scale * I_total * lam (=2*pi*fn)
        k2=0.2 * (0.00202688 + 0.045970) * 71530.0, # k2_scale * I_total * (lam * lam)
        fn=2.0
    ),
    MotorControlCfg(
        joint_name="R_toe_joint",
        joint_index=9,
        b=0.0,
        tau_c=0.0,
        I_total=0.00202688 + 0.045970,
        tau_s_max=25.0,
        k1=1.1 * (0.00202688 + 0.045970) * math.sqrt(3000.0), # k1_scale * I_total * lam (=2*pi*fn)
        k2=0.2 * (0.00202688 + 0.045970) * 3000.0, # k2_scale * I_total * (lam * lam)
        fn=2.0
    ),
]


class STAController(nn.Module):
    """
    tau = tau_f + tau_s
      tau_f = b*qd + tau_c*tanh(qd/v_eps)

    s = e_dot + lambda*e
    v = v - k2*dt*sign(s)
    tau_s_raw = v - k1*sqrt(|s|)*sign(s) + M * lambda * e_dot
    tau_s = tau_s_max * tanh(tau_s_raw / tau_s_max)   (soft saturation)
    """

    def __init__(
        self,
        cfgs: Sequence[MotorControlCfg],
        dof: int = 10,
        v_eps: float = 0.1,
        sta_eps: float = 1e-6,
        sign_deadzone: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.dof = int(dof)
        self.v_eps = float(v_eps)
        self.inv_v_eps = float(1.0 / v_eps)
        self.sta_eps = float(sta_eps)
        self.sign_deadzone = float(sign_deadzone)

        b = torch.zeros(self.dof, device=device, dtype=dtype)
        tau_c = torch.zeros(self.dof, device=device, dtype=dtype)
        lam = torch.zeros(self.dof, device=device, dtype=dtype)
        k1 = torch.zeros(self.dof, device=device, dtype=dtype)
        k2 = torch.zeros(self.dof, device=device, dtype=dtype)
        tau_s_max = torch.ones(self.dof, device=device, dtype=dtype)
        I_total_tensor = torch.ones(self.dof, device=device, dtype=dtype)

        names = [None] * self.dof
        for c in cfgs:
            j = int(c.joint_index)
            if not (0 <= j < self.dof):
                raise ValueError(f"joint_index {j} out of range for {c.joint_name}")
            if names[j] is not None:
                raise ValueError(f"duplicate cfg for joint_index {j}")
            names[j] = c.joint_name

            b[j] = float(c.b)
            tau_c[j] = float(c.tau_c)
            tau_s_max[j] = float(c.tau_s_max)
            I_total_tensor[j] = float(c.I_total)

            # lambda
            if c.lambda_ is not None:
                lam_j = float(c.lambda_)
            else:
                if c.fn is None:
                    lam_j = 0.0
                else:
                    lam_j = float(2.0 * 3.141592653589793 * float(c.fn))
            lam[j] = lam_j

            # k1/k2
            if (c.k1 is not None) and (c.k2 is not None):
                k1[j] = float(c.k1)
                k2[j] = float(c.k2)
            else:
                raise ValueError(
                    f"{c.joint_name}: provide k1,k2 OR provide fn/lambda_ for auto gain init."
                )
           

        missing = [i for i, n in enumerate(names) if n is None]
        if missing:
            raise ValueError(f"Missing cfgs for joint indices: {missing}")

        self.register_buffer("b", b)
        self.register_buffer("tau_c", tau_c)
        self.register_buffer("lam", lam)
        self.register_buffer("k1", k1)
        self.register_buffer("k2", k2)
        self.register_buffer("tau_s_max", tau_s_max)
        self.register_buffer("inv_tau_s_max", 1.0 / (tau_s_max + 1e-12))
        self.register_buffer("M", I_total_tensor)

        # state: v (N,dof)
        self.register_buffer("_v", torch.zeros(0, self.dof, device=device, dtype=dtype), persistent=False)


    @torch.no_grad()
    def reset(self, N: int, device=None, dtype=None):
        if device is None:
            device = self.b.device
        if dtype is None:
            dtype = self.b.dtype
        self._v = torch.zeros(N, self.dof, device=device, dtype=dtype)

    @torch.no_grad()
    def step(
        self,
        q: torch.Tensor,          # (N,dof)
        qd: torch.Tensor,         # (N,dof)
        q_ref: torch.Tensor,      # (N,dof)
        qd_ref: torch.Tensor | None = None,
        dt: float = 0.002,
        aw_strength: float = 1.0, # 1.0 = full gating, 0.0 = no anti-windup
    ) -> torch.Tensor:
        N = q.shape[0]
        if self._v.numel() == 0 or self._v.shape[0] != N or self._v.device != q.device or self._v.dtype != q.dtype:
            self.reset(N, device=q.device, dtype=q.dtype)

        # friction model
        # tau_f = self.b * qd + self.tau_c * torch.tanh(qd * self.inv_v_eps)

        # s = e_dot + lam*e
        if qd_ref is None:
            e_dot = -qd
        else:
            e_dot = qd_ref - qd 
        e = q_ref - q
        s = e_dot + self.lam * e

        abs_s = s.abs()
        # sign_s = s.sign()
        # sign_s.mul_((abs_s > self.sign_deadzone).to(sign_s.dtype))
        sign_s = s / (abs_s + self.sign_deadzone)  # smooth sign with deadzone around zero

        # compute STA static part
        sqrt_abs_s = torch.sqrt(abs_s + self.sta_eps)
        a = self.k1 * sqrt_abs_s * sign_s  # a = k1*sqrt(|s|)*sign(s)
        # a = self.k1 * abs_s * sign_s
        a += self.M * self.lam * e_dot  # + M*lam*e_dot
        # a += self.k2 * s

        #  Anti-windup for STA integrator (v)
        # Predict raw torque BEFORE updating v:
        # tau_s_raw_pre = v + a
        tau_s_raw_pre = self._v + a

        # Soft-saturation output uses tanh(x)
        x = tau_s_raw_pre * self.inv_tau_s_max
        y = torch.tanh(x)                  

        # Gate = derivative of tanh: sech^2(x) = 1 - tanh(x)^2
        g = 1.0 - y * y                  
        if aw_strength != 1.0:
            g.mul_(aw_strength).add_(1.0 - aw_strength)

        # v = v + (k2*dt)*sign(s) * g
        self._v.add_(sign_s * (self.k2 * dt) * g)
        # self._v.add_(sign_s * (self.k2 * dt))

        # # tau_s with updated v
        tau_s_raw = self._v + a
        # tau_s_raw = a
        tau_s = self.tau_s_max * torch.tanh(tau_s_raw * self.inv_tau_s_max)

        return tau_s

    def forward(self, q, qd, q_ref, qd_ref=None, dt: float = 0.002):
        # wrapping around forward because compile can optimize this.
        return self.step(q, qd, q_ref, qd_ref, float(dt), 1.0)