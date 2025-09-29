from curses.ascii import ctrl
import math
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import torch

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
    EventCfg,
    ObservationsCfg,
)       
from isaaclab.managers import TerminationTermCfg as DoneTerm

from isaaclab.utils import configclass
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    EventTermCfg,
)
from isaaclab.utils.noise import AdditiveUniformNoiseCfg, AdditiveGaussianNoiseCfg
from isaaclab.envs.mdp.actions import joint_actions
##
# User defined configs
##
from isaac_lab_actuator_dynamic.assets import LEGACTUATORDYNAMIC_CFG, LEGACTUATORDYNAMIC_2_CFG, LEGWALKING_CFG
from . import mdp as custom_mdp

import torch

import torch
from typing import Optional, Tuple
import torch.nn as nn

# ---- Utilities --------------------------------------------------------------

def _normalize_ba(b: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if a[0] != 1.0:
        b = b / a[0]
        a = a / a[0]
    return b, a

# ---- Optimized DF2T biquad --------------------------------------------------

class FastSecondOrderLPF(nn.Module):
    """
    Optimized DF2T biquad:
        y[n]  = b0*x[n] + z1
        z1'   = b1*x[n] - a1*y[n] + z2
        z2'   = b2*x[n] - a2*y[n]

    Optimizations:
      - 2 state tensors total (z1, z2) and in-place updates (no new tensors each step)
      - Works on arbitrary shapes [...], e.g. [N,D], [B,C,H,W] (broadcasted coeffs)
      - reset() supports preallocation to avoid first-call shape checks
      - forward_seq() processes [T, ...] in one compiled/scripted region (no Python loop cost)
    """

    def __init__(self, b, a):
        super().__init__()
        b = torch.as_tensor(list(b), dtype=torch.float32)
        a = torch.as_tensor(list(a), dtype=torch.float32)
        b, a = _normalize_ba(b, a)

        self.register_buffer("b", b)   # [b0, b1, b2]
        self.register_buffer("a", a)   # [1, a1, a2]
        self._z = None                 # state: [2, *shape] -> z1, z2

    @torch.no_grad()
    def reset(
        self,
        shape: Optional[Tuple[int, ...]] = None,
        value: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Clear or preallocate states.
        - shape=None: clear (recreated on first forward)
        - shape=... : allocate [2, *shape] and fill with `value`
        """
        if shape is None:
            self._z = None
            return
        device = device or self.b.device
        dtype  = dtype  or torch.float32
        self._z = torch.full((2, *shape), value, device=device, dtype=dtype)

    def _ensure_state(self, x: torch.Tensor):
        tgt = (2, *x.shape)
        z = self._z
        if (z is None or z.shape != tgt or z.device != x.device or z.dtype != x.dtype):
            # Single alloc; no clones per step
            self._z = torch.zeros(tgt, device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Single step on shape [...]. """
        self._ensure_state(x)

        z1 = self._z[0]
        z2 = self._z[1]

        b0, b1, b2 = self.b
        _,  a1, a2 = self.a

        # y = b0*x + z1
        y = b0 * x + z1

        # In-place state update without building autograd history
        with torch.no_grad():
            # z1 <- b1*x - a1*y + z2
            # z2 <- b2*x - a2*y
            # (use addcmul-like fused ops when beneficial; PyTorch often fuses anyway)
            z1.mul_(0).add_(b1 * x).add_(-a1 * y).add_(z2)
            z2.mul_(0).add_(b2 * x).add_(-a2 * y)

        return y

import torch
import torch.nn as nn
from typing import Optional

class INDI_BatchedPendulum_Fast(nn.Module):
    """
    Faster INDI for B x J pendulum joints.
      Model: I * qdd + m g l * sin(q) = tau  (ignore velocity Jacobian B)
      Jacobians: G = 1/I,  A = -(m g l / I) * cos(q_prev)
    Inputs each tick: q_ref, q_meas, qd_meas, qdd_meas  -> all (B,J)
    Outputs: tau, tau_pd, tau_ff  -> all (B,J)
    """
    def __init__(self, n_envs=4096, n_joints=10, Ts=1e-3,
                 joint_bw_hz=5.0,
                 torque_limit=100.0, torque_rate_limit=2000.0,
                 use_AB: bool = False,
                 use_model_jacobians: bool = True):
        super().__init__()
        self.B, self.J, self.Ts = n_envs, n_joints, Ts
        self.use_AB = use_AB
        self.use_model_jacobians = use_model_jacobians

        # Gains for ζ=1 from target bandwidth
        w = 2.0 * torch.pi * torch.as_tensor(joint_bw_hz, dtype=torch.float32)
        self.register_buffer('Kp', torch.full((self.B, self.J), float(w * w)))
        self.register_buffer('Kd', torch.full((self.B, self.J), float(2.0 * w)))

        # States (histories)
        self.register_buffer('tau_prev', torch.zeros(self.B, self.J))
        self.register_buffer('q_prev',   torch.zeros(self.B, self.J))
        self.register_buffer('qd_prev',  torch.zeros(self.B, self.J))
        self.register_buffer('qdd_prev', torch.zeros(self.B, self.J))

        # Learned fallbacks (unused in model path but kept API)
        self.register_buffer('Ghat', torch.ones(self.B, self.J))
        self.register_buffer('Ahat', torch.zeros(self.B, self.J))

        # Limits
        self.torque_limit      = float(torque_limit)
        self.torque_rate_limit = float(torque_rate_limit)

        # Physical params
        self.register_buffer('I',  torch.ones(self.B, self.J))
        self.register_buffer('m',  torch.ones(self.B, self.J))
        self.register_buffer('g',  torch.full((self.B, self.J), 9.81))
        self.register_buffer('l',  torch.ones(self.B, self.J))
        # Precompute mgl/I (avoids div per tick)
        self.register_buffer('mgl_over_I', torch.ones(self.B, self.J))

        # -------- Scratch buffers (reused every step; no allocations) --------
        self.register_buffer('_pd_term',   torch.zeros(self.B, self.J))
        self.register_buffer('_delta_tau', torch.zeros(self.B, self.J))
        self.register_buffer('_tau_tmp',   torch.zeros(self.B, self.J))
        self.register_buffer('_margin',    torch.zeros(self.B, self.J))
        self.register_buffer('_step_need', torch.zeros(self.B, self.J))

        # Caps (kept as python floats; fused in kernel)
        self.a_ff_cap = 1200.0
        self.a_pd_cap = 1000.0

    # ---------------- Utilities ----------------
    def _recompute_mgl_over_I(self):
        # Avoid divide-by-zero; clamp once
        I_safe = torch.clamp(self.I, min=1e-9)
        self.mgl_over_I.copy_( (self.m * self.g * self.l) / I_safe )

    def set_limits(self, torque_limit: float = 100.0, torque_rate_limit: float = 2000.0):
        self.torque_limit      = float(torque_limit)
        self.torque_rate_limit = float(torque_rate_limit)

    def set_critically_damped(self, f_hz):
        if not torch.is_tensor(f_hz):
            f_hz = torch.full((self.B, self.J), float(f_hz), device=self.Kp.device, dtype=self.Kp.dtype)
        elif f_hz.ndim == 1:
            f_hz = f_hz.view(1, -1).expand(self.B, self.J).to(self.Kp.device, self.Kp.dtype)
        else:
            f_hz = f_hz.to(self.Kp.device, self.Kp.dtype)
        w = 2.0 * torch.pi * f_hz
        self.Kp.copy_(w * w)
        self.Kd.copy_(2.0 * w)

    def set_gains(self, kp: float | torch.Tensor, kd: float | torch.Tensor):
        dev = self.Kp.device; shape = (self.B, self.J)
        def expand(x):
            if torch.is_tensor(x):
                x = x.to(dev, dtype=self.Kp.dtype)
                if x.ndim == 1 and x.shape[0] == self.J:
                    return x.view(1, -1).expand(shape)
                elif x.shape == shape:
                    return x
                else:
                    raise ValueError(f"Invalid gain shape {tuple(x.shape)}")
            else:
                return torch.full(shape, float(x), device=dev, dtype=self.Kp.dtype)
        self.Kp.copy_(expand(kp))
        self.Kd.copy_(expand(kd))

    def set_params(self, I: float | torch.Tensor, m: float | torch.Tensor,
                   g: float | torch.Tensor, l: float | torch.Tensor):
        dev = self.I.device; shape = (self.B, self.J)
        def expand(x):
            if torch.is_tensor(x):
                x = x.to(dev, dtype=torch.float32)
                if x.ndim == 1 and x.shape[0] == self.J:
                    return x.view(1, -1).expand(shape).clone()
                elif x.shape == shape:
                    return x.clone()
                else:
                    raise ValueError(f"Invalid param shape {tuple(x.shape)}; need scalar, (J,), or (B,J).")
            else:
                return torch.full(shape, float(x), device=dev, dtype=torch.float32)
        self.I.copy_(expand(I))
        self.m.copy_(expand(m))
        self.g.copy_(expand(g))
        self.l.copy_(expand(l))
        self._recompute_mgl_over_I()  # refresh cached ratio

    @torch.no_grad()
    def reset(self, mask: Optional[torch.Tensor] = None):
        if mask is None:
            self.tau_prev.zero_()
            self.q_prev.zero_()
            self.Ghat.fill_(1.0)
            self.Ahat.zero_()
        else:
            mask = mask.view(-1, 1).expand(-1, self.J).to(dtype=torch.bool)
            self.tau_prev.masked_fill_(mask, 0.0)
            self.q_prev.masked_fill_(mask,   0.0)
            self.Ghat.masked_fill_(mask, 1.0)
            self.Ahat.masked_fill_(mask, 0.0)

    # ---------------- Control tick (optimized) ----------------
    @torch.no_grad()
    def step(self, q_ref: torch.Tensor, q_meas: torch.Tensor,
             qd_meas: torch.Tensor, qdd_meas: torch.Tensor):
        """
        One control tick. All tensors (B,J) on same device/dtype.
        Returns: tau, tau_pd, tau_ff  (each (B,J))
        """
        # pd_term = Kd*(0 - qd) + Kp*(q_ref - q)
        # Use scratch buffer _pd_term
        self._pd_term.mul_(0.0)\
            .add_(self.Kp, alpha=1.0).mul_(q_ref)\
            .addcmul_(self.Kp, -q_meas, value=1.0)\
            .addcmul_(self.Kd, -qd_meas, value=1.0)
        # Explanation:
        # _pd = Kp*q_ref - Kp*q_meas - Kd*qd_meas

        if self.use_model_jacobians:
            # Ginv = I (since G = 1/I). Precomputed mgl/I.
            # Optional A*Δq subtraction: pd_term -= A_model * (q - q_prev)
            if self.use_AB:
                # A_model = -(mgl/I)*cos(q_prev)
                A_model = -self.mgl_over_I * torch.cos(self.q_prev)
                # _pd_term -= A_model * dq
                self._pd_term.addcmul_(A_model, (q_meas - self.q_prev), value=-1.0)
            Ginv = self.I  # broadcasted multiply later
        else:
            # Fallback learned (kept for API symmetry)
            Gabs = torch.clamp(self.Ghat.abs(), min=1e-3, max=1e3)
            Ginv = 1.0 / Gabs
            if self.use_AB:
                self._pd_term.addcmul_(self.Ahat, (q_meas - self.q_prev), value=-1.0)

        # INDI increments (qdd_r = 0, qd_r = 0)
        # ff_inc = - qdd_hat ;  pd_inc = _pd_term
        ff_inc = (-qdd_meas).clamp_(-self.a_ff_cap, self.a_ff_cap)
        pd_inc = self._pd_term.clamp_(-self.a_pd_cap, self.a_pd_cap)

        # delta_tau = Ginv*(ff_inc + pd_inc)
        self._delta_tau.mul_(0.0).add_(ff_inc).add_(pd_inc)
        self._delta_tau.mul_(Ginv)

        # tau_desired = tau_prev + delta_tau
        self._tau_tmp.copy_(self.tau_prev).add_(self._delta_tau)

        # Rate limit first
        max_step = self.torque_rate_limit * self.Ts
        tau_rate = torch.clamp(self._tau_tmp,
                               min=self.tau_prev - max_step,
                               max=self.tau_prev + max_step)

        # Soft anti-windup scaling near ±limit
        self._margin.copy_(self.torque_limit).sub_(self.tau_prev.abs()).clamp_min_(1e-6)
        self._step_need.copy_(tau_rate).sub_(self.tau_prev).abs_().clamp_min_(1e-9)
        alpha = torch.clamp(self._margin / self._step_need, max=1.0)
        self._tau_tmp.copy_(self.tau_prev).addcmul_(alpha, (tau_rate - self.tau_prev))

        # Hard clamp last
        tau = torch.clamp(self._tau_tmp, -self.torque_limit, self.torque_limit)

        # Outputs before limits:
        tau_pd = self._delta_tau - (Ginv * ff_inc)     # only PD increment
        tau_ff_raw = self.tau_prev + (Ginv * ff_inc)   # FF pre-limit

        # Shift histories (in-place, no graph)
        self.tau_prev.copy_(tau)
        self.q_prev.copy_(q_meas)

        return tau, tau_pd, tau_ff_raw


class CustomJointPositionAction(joint_actions.JointPositionAction):
    def __init__(self, cfg, env):
        # initialize the action term
        super().__init__(cfg, env)
        
        
        N, Ts = 10, 0.001
        self.ctrl = INDI_BatchedPendulum_Fast(
            n_envs=self.num_envs, n_joints=self.action_dim, Ts=Ts,
            joint_bw_hz=15.0,
            torque_limit=150.0, torque_rate_limit=1000.0,
            use_AB=True
        ).to(self.device)

        J = self.action_dim
        I_j = torch.tensor([0.016, 0.016, 0.014, 0.014, 0.12, 0.12, 0.024, 0.024, 0.0025, 0.0025],
                        dtype=torch.float32, device=self.device)
        m_j = torch.tensor([1.53, 1.53, 1.33, 1.33, 4.91, 4.91, 1.5, 1.5, 0.52014, 0.52014],
                        dtype=torch.float32, device=self.device)
        l_j = torch.tensor([0.094, 0.094, 0.089, 0.089, 0.088, 0.088, 0.0777, 0.0777, 0.042, 0.042],
                   dtype=torch.float32, device=self.device)
        g_j = torch.full((J,), 9.81, device=self.device)              # gravity
        self.ctrl.set_params(I=I_j, m=m_j, g=g_j, l=l_j)

        # self.ctrl.set_critically_damped(2.1)  # set all joints to 2 Hz natural frequency
        self.ctrl.Kp.fill_(100.0)
        self.ctrl.Kd.fill_(20.0)

        # filter of action and estimated joint acc
        b = [0.26266595, 0.52533190, 0.26266595]
        a = [1.0, -0.24814856, 0.29881237]
        self.action_filter = FastSecondOrderLPF(b, a).to(self.device)
        self.action_filter.reset(shape=(self.num_envs, self.action_dim), value=0.0, device=self.device)
        self.filtered_actions = torch.zeros_like(self.processed_actions)

        self.joint_acc_filter = FastSecondOrderLPF(b, a).to(self.device)
        self.joint_acc_filter.reset(shape=(self.num_envs, self.action_dim), value=0.0, device=self.device)
        self.filtered_joint_acc = torch.zeros_like(self.processed_actions)

        b = [0.00838838, 0.01677676, 0.00838838]
        a = [1.0, -1.66417053, 0.69772405]
        self.action_filter_2 = FastSecondOrderLPF(b, a).to(self.device)
        self.action_filter_2.reset(shape=(self.num_envs, self.action_dim), value=0.0, device=self.device)
        self.filtered_actions_2 = torch.zeros_like(self.processed_actions)

        self.counter = 0
        self.is_first = True
        self.max_counter = 5.0 / 0.001
        self.my_action = torch.zeros_like(self.processed_actions)

    def apply_actions(self):
        # if self.is_first:
        #     first_q_meas = torch.zeros_like(self.processed_actions)
        #     self.my_action = self._asset.data.joint_pos.clone()

        #     q_cmd = torch.zeros_like(self.processed_actions)
        #     q_cmd[:,4] = 0.2; q_cmd[:,5] = 0.2
        #     q_cmd[:,6] = -0.34; q_cmd[:,7] = -0.34
        #     q_cmd[:,8] = 0.2; q_cmd[:,9] = 0.2

        #     self.is_first = False

        # alpha = 0.9999
        # self.counter += 1
        # if self.counter >= self.max_counter:
        #     self.counter = self.max_counter
        # self.my_action = alpha * self.my_action + (1-alpha) * q_cmd
        
        self.filtered_actions_2 = self.action_filter_2(self.processed_actions).clone()

        # get q_meas, qd_meas
        q_meas = self._asset.data.joint_pos
        qd_meas = self._asset.data.joint_vel
        self.filtered_joint_acc = self._asset.data.joint_acc     
        # self.filtered_joint_acc = self.joint_acc_filter(self.filtered_joint_acc).clone()    
        self.filtered_actions, _, _ = self.ctrl.step(self.processed_actions, q_meas, qd_meas, self.filtered_joint_acc)
        # self.filtered_actions = self.action_filter(self.filtered_actions).clone()

      
        self._asset.set_joint_effort_target(self.filtered_actions, joint_ids=self._joint_ids)
        # self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids) -> None:
        self._raw_actions[env_ids] = 0.0
        self.filtered_actions[env_ids] = 0.0
        self.ctrl.reset(env_ids)
        self.joint_acc_filter.reset(shape=(self.num_envs, self.action_dim), value=0.0, device=self.device)
        self.action_filter.reset(shape=(self.num_envs, self.action_dim), value=0.0, device=self.device)
        self.action_filter_2.reset(shape=(self.num_envs, self.action_dim), value=0.0, device=self.device)
        self.is_first = True
        self.counter = 0
        self.my_action = torch.zeros_like(self.processed_actions)

class Custom2JointPositionAction(joint_actions.JointPositionAction):
    def __init__(self, cfg, env):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        # if cfg.use_default_offset:
        #     self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        #     self._offset[:, 4] = 0.51; self._offset[:, 5] = 0.51
        #     self._offset[:, 6] = -0.85; self._offset[:, 7] = -0.85
        #     self._offset[:, 8] = 0.6; self._offset[:, 9] = 0.6

        self.filtered_actions = torch.zeros_like(self.processed_actions)

    def apply_actions(self):

        alpha = 0.0
        self.filtered_actions = alpha * self.filtered_actions + (1- alpha) * self.processed_actions
        self._asset.set_joint_position_target(self.filtered_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids) -> None:
        self.filtered_actions[env_ids] = 0.0



@configclass
class Actions2PlayCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(class_type=CustomJointPositionAction, asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=False)
    # joint_pos = mdp.JointPositionActionCfg(class_type=Custom2JointPositionAction, asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True)

@configclass
class WalkingRobotObservationsCfg(ObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""
        # base state
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.05), clip=(-100.0, 100.0), scale=0.25)
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.025), clip=(-100.0, 100.0), scale=1.0)
        
        # velocity command
        velocity_commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # joint state
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=1.0)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.3), clip=(-100.0, 100.0), scale=0.05)
        
        # last action
        actions = ObservationTermCfg(func=mdp.last_action)

        # gaits
        phase = ObservationTermCfg(func=custom_mdp.get_phase)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True


    @configclass
    class CriticCfg(ObservationGroupCfg):
        """Observations for policy group."""
        # base state
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel)
        proj_gravity = ObservationTermCfg(func=mdp.projected_gravity)
        heights = ObservationTermCfg(func=mdp.height_scan,params={"sensor_cfg": SceneEntityCfg("height_scanner")})
        robot_base_pos = ObservationTermCfg(func=mdp.root_pos_w)
        robot_base_quat = ObservationTermCfg(func=mdp.root_quat_w)
        robot_base_lin_vel = ObservationTermCfg(func=mdp.root_lin_vel_w)
        robot_base_ang_vel = ObservationTermCfg(func=mdp.root_ang_vel_w)

        # velocity command
        velocity_commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # joint state
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel)

        # last action
        last_action = ObservationTermCfg(func=mdp.last_action)

        # contact state and phase
        contact_state = ObservationTermCfg(
            func=custom_mdp.contact_state, 
            params={
                "sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=["L_toe", "R_toe"]),
                "asset_cfg": SceneEntityCfg("robot", body_names=["L_toe", "R_toe"]),
            }
        )
        phase = ObservationTermCfg(func=custom_mdp.get_phase)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class DebugCfg(ObservationGroupCfg):
        """Observations for debug group."""
        joint_torque = ObservationTermCfg(func=custom_mdp.joint_torque)
        base_height = ObservationTermCfg(func=mdp.base_pos_z)
        joint_acc = ObservationTermCfg(func=custom_mdp.joint_acc)
        joint_pos = ObservationTermCfg(func=mdp.joint_pos)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel)
        # joint_cmd_pos = ObservationTermCfg(func=custom_mdp.joint_pos_and_cmd)
        # joint_cmd_pos_error = ObservationTermCfg(func=custom_mdp.joint_vel_and_cmd_error)


        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg() 
    critic: CriticCfg = CriticCfg()
    #! Just for debugging
    debug: DebugCfg = DebugCfg()
    

@configclass 
class WalkingRobotEventCfg(EventCfg):
    update_phase = EventTermCfg(
        func=custom_mdp.update_phase,
        mode="interval",
        interval_range_s=(0.0, 0.0),
    )

    def __post_init__(self):
        super().__post_init__() #type: ignore

        ''' #!Domain randomization setup
            The default domain randomization setup includes:
            1. physic material
            2. add base mass
            3. base external force torque
            4. reset base
            5. reset robot joint
            6. push robot

        '''
        self.physics_material.params["dynamic_friction_range"] = [0.9, 1.25]
        self.physics_material = None
        # self.base_com = None
        # self.add_base_mass.params["mass_distribution_params"] = [-5.0, 10.0]
        self.add_base_mass = None
        self.push_robot.params = {
            "velocity_range": {
                "x": [-1.0, 1.0],
                "y": [-1.0, 1.0],
            }
        }
        self.push_robot.interval_range_s = (2.5, 2.5)
        self.base_external_force_torque = None 
        self.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        # # self.reset_robot_joints.func = mdp.reset_joints_by_offset
        self.reset_robot_joints.params = {
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        }

@configclass
class WalkingRobotRewardCfg:
    track_lin_vel_xy_exp = RewardTermCfg(
        func=mdp.track_lin_vel_xy_exp, weight=3.0, params={"command_name": "base_velocity", "std": math.sqrt(0.2)}
    )
    track_ang_vel_z_exp = RewardTermCfg(
        func=mdp.track_ang_vel_z_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.2)}
    )
    is_alive = RewardTermCfg(
        func=mdp.is_alive,
        weight=1.0,
    )
    feet_schedule_contact = RewardTermCfg(
        func=custom_mdp.feet_schedule_contact_with_cmd,
        weight=1.2,
        params={"sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=["L_toe", "R_toe"])},
    )

    # penalty terms
    dof_torques_l2 = RewardTermCfg(func=custom_mdp.weighted_joint_torques_l2, weight=-8.0e-5)
    base_height_l2 = RewardTermCfg(
        func=mdp.base_height_l2,
        weight=-20.0,
        params={"target_height": 0.78},
    )
    dof_vel = RewardTermCfg(
        func=mdp.joint_vel_l2,
        weight=-1e-3,
    )
    # dof_acc_l2 = RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-5)
    lin_vel_z_l2 = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05)
    action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.03)
    dof_pos_limits = RewardTermCfg(func=mdp.joint_pos_limits, weight=-2.0)
    action_limits = RewardTermCfg(func=custom_mdp.action_limits, weight=-2.0)
    undesired_contacts = RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_hip", ".*_hip2", "base"]), "threshold": 10.0},
    )
    action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.4)
    flat_orientation_l2 = RewardTermCfg(func=mdp.flat_orientation_l2, weight=-10.0)
    action_norm = RewardTermCfg(
        func=mdp.action_l2,
        weight=-0.001,
    )
    joint_deviation = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_hip2_joint"])},
    )
    # feet_distance = RewardTermCfg(
    #     func=custom_mdp.feet_distance,
    #     weight=-100,
    #     params={"min_feet_distance": 0.115,"feet_links_name": ["foot_[RL]_Link"]}
    # )

    feet_height = RewardTermCfg(
        func=custom_mdp.feet_height,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=["L_toe", "R_toe"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["L_toe", "R_toe"]),
        },
    )

    stand_still_contact = RewardTermCfg(
        func=custom_mdp.stand_still_contact,
        weight=-0.7,
        params={
            "sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=["L_toe", "R_toe"]),
        },
    )

    stand_still = RewardTermCfg(
        func=custom_mdp.stand_still,
        weight=-0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot")
        },
    )

class CustomUniformVelocityCommand(mdp.UniformVelocityCommand):
    """Custom uniform velocity command generator configuration."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self.metrics["num_standing_envs"] = torch.zeros(self.num_envs, device=self.device)

    def _update_metrics(self):
        super()._update_metrics()
        # -- metrics
        self.metrics["num_standing_envs"] = self.is_standing_env.float()

@configclass
class WalkingRobotCommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        class_type=CustomUniformVelocityCommand,
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.35,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.0, 2.0), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-1.0, 1.0)
        ),
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"asset_cfg": SceneEntityCfg("robot"), "minimum_height": 0.3},)

@configclass
class WalkingRobotEnvCfg(LocomotionVelocityRoughEnvCfg):
    observations: WalkingRobotObservationsCfg = WalkingRobotObservationsCfg()
    rewards: WalkingRobotRewardCfg = WalkingRobotRewardCfg()
    events: WalkingRobotEventCfg = WalkingRobotEventCfg()
    commands: WalkingRobotCommandsCfg = WalkingRobotCommandsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    actions: Actions2PlayCfg = Actions2PlayCfg()

    def __post_init__(self):
        super().__post_init__()

        ''' #!Terrain setup'''
        # self.scene.terrain.terrain_generator = custom_mdp.TERRAINS_CFG
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator.curriculum = False #type: ignore
        self.curriculum.terrain_levels = None
        self.sim.episode_length_s = 5.0
        self.sim.dt = 0.001
        self.decimation = 20

        ''' #!Action setup
            The default action space setup includes:
            1. joint position command
        '''
        # q_cmd = q_default + scale * network_output
        self.actions.joint_pos.scale = 1.0
        
        self.scene.robot = LEGWALKING_CFG.replace(prim_path="/World/envs/env_.*/Robot") #type: ignore 
        # self.scene.height_scanner = None #type: ignore

        # self.sim.dt = 0.001
        # self.decimation = 10
        step_dt = self.sim.dt * self.decimation
        self.events.update_phase.interval_range_s = (step_dt, step_dt)

        ''' #!Termination setup
            The default termination setup includes:
            1. base contact
            2. time out
        '''
        #* Remain the default termination setup

@configclass
class WalkingRobotEnvPLayCfg(WalkingRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # self.sim.render_interval = 8
        self.episode_length_s = 20.0
        self.sim.dt = 0.001
        self.decimation = 20
        self.sim.render_interval = self.decimation

        # self.observations.policy.enable_corruption = False

        # self.events.add_base_mass = 
        self.events.base_com.params = {
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.2, -0.2), "y": (-0.00, 0.00), "z": (-0.00, 0.00)},
        }
        self.events.base_com = None #type: ignore
        self.events.base_external_force_torque = None
        self.events.push_robot = None #type: ignore
        # self.events.reset_base = None #type: ignore
        # self.events.reset_robot_joints = None #type: ignore
        self.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator.curriculum = False #type: ignore
        self.curriculum.terrain_levels = None #type: ignore

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.00, -0.00)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
   

        # self.viewer.asset_name = "robot"
        # self.viewer.origin_type = "asset_root"
        # self.viewer.eye = (0.0, 5.0, 2.0)


        # self.sim.use_fabric = False
        # self.sim.device = "cpu"
        # self.actions.joint_pos.scale = {"R_toe_joint": 0.001}
        # self.actions.joint_pos = Actions2PlayCfg()

 


    



    