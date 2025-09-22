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


# ---------- Utilities ----------
def _lpf(old: torch.Tensor, new: torch.Tensor, alpha: float) -> torch.Tensor:
    """Exponential moving average."""
    return (1.0 - alpha) * old + alpha * new


# ---------- Command filter: q_ref -> (q_r, qd_r, qdd_r) ----------
class CmdFilter2nd(nn.Module):
    """Critically-damped 2nd-order reference shaper (vectorized B x J)."""
    def __init__(self, n_envs: int, n_joints: int, Ts: float, fc_hz: float = 20.0):
        super().__init__()
        self.B, self.J, self.Ts = n_envs, n_joints, Ts
        w = 2.0 * torch.pi * torch.as_tensor(fc_hz, dtype=torch.float32)
        self.register_buffer('wc', torch.full((self.B, self.J), w, dtype=torch.float32))
        self.register_buffer('x0', torch.zeros(self.B, self.J))  # q_r
        self.register_buffer('x1', torch.zeros(self.B, self.J))  # qd_r
        self.register_buffer('x2', torch.zeros(self.B, self.J))  # qdd_r

    def forward(self, q_ref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wc, Ts = self.wc, self.Ts
        e0 = q_ref - self.x0
        dx2 = wc * wc * e0 - 2.0 * wc * self.x1 - wc * wc * self.x0
        self.x2 = self.x2 + Ts * dx2
        self.x1 = self.x1 + Ts * self.x2
        self.x0 = self.x0 + Ts * self.x1
        return self.x0, self.x1, self.x2  # (q_r, qd_r, qdd_r)


# ---------- 1st-order Levant (on velocity -> acceleration) ----------
class LevantOrder1(nn.Module):
    """
    Levant differentiator (order-1) on measured velocity: qd -> qdd_hat.
    Vectorized over (B, J). Uses boundary-layer 'sat' regularization.
    """
    def __init__(self, n_envs: int, n_joints: int, Ts: float, fd_hz: float = 20.0,
                 k1_scale: float = 1.5, k2_scale: float = 1.1, eps: float = 1e-3,
                 a_guard: float = 5e4):
        super().__init__()
        self.B, self.J, self.Ts = n_envs, n_joints, Ts
        w = 2.0 * torch.pi * torch.as_tensor(fd_hz, dtype=torch.float32)
        # Keep gains as floats; states are tensors on the module's device
        self.k1 = float(k1_scale * torch.sqrt(w))
        self.k2 = float(k2_scale * w)
        self.eps = float(eps)
        self.a_guard = float(a_guard)
        self.register_buffer('z0', torch.zeros(self.B, self.J))  # tracks y=qd
        self.register_buffer('z1', torch.zeros(self.B, self.J))  # tracks qdd

    @staticmethod
    def _sat(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, -1.0, 1.0)

    def forward(self, qd_meas: torch.Tensor) -> torch.Tensor:
        e = self.z0 - qd_meas
        se = self._sat(e / self.eps)
        dz0 = self.z1 - self.k1 * torch.sqrt(e.abs() + 1e-12) * se
        dz1 = - self.k2 * se
        self.z0 = self.z0 + self.Ts * dz0
        self.z1 = torch.clamp(self.z1 + self.Ts * dz1, -self.a_guard, self.a_guard)
        return self.z1  # qdd_hat  (B,J)


# ---------- INDI (batched) with pendulum model Jacobians ----------
class INDI_BatchedPendulum(nn.Module):
    """
    INDI for B x J pendulum joints:
      Model: I * qdd + m g l * sin(q) = tau  (ignore velocity Jacobian B)
      Jacobians: G = 1/I,  A = -(m g l / I) * cos(q_{t-1})
    Inputs each tick: q_ref, q_meas, qd_meas  -> all (B,J)
    Outputs: tau, tau_pd, tau_ff  -> all (B,J)
    """
    def __init__(self, n_envs=4096, n_joints=10, Ts=1e-3,
                 joint_bw_hz=5.0, fc_hz=20.0, fd_hz=20.0,
                 torque_limit=100.0, torque_rate_limit=2000.0,
                 use_AB: bool = False,        # if True, subtract learned Ahat*dq (ignored if use_model_jacobians=True)
                 use_model_jacobians: bool = True,  # if True, use A = -(m g l / I) cos(q_prev), Ginv = I
                 ):
        super().__init__()
        self.B, self.J, self.Ts = n_envs, n_joints, Ts
        self.use_AB = use_AB
        self.use_model_jacobians = use_model_jacobians

        # Error-law gains from target bandwidth (ζ=1 baseline)
        w = 2.0 * torch.pi * torch.as_tensor(joint_bw_hz, dtype=torch.float32)
        self.register_buffer('Kp', torch.full((self.B, self.J), float(w * w)))
        self.register_buffer('Kd', torch.full((self.B, self.J), float(2.0 * w)))

        # States (histories)
        self.register_buffer('tau_prev', torch.zeros(self.B, self.J))
        self.register_buffer('q_prev',   torch.zeros(self.B, self.J))
        self.register_buffer('qd_prev',  torch.zeros(self.B, self.J))
        self.register_buffer('qdd_prev', torch.zeros(self.B, self.J))

        # Learned Jacobians (optional fallback if use_model_jacobians=False)
        self.register_buffer('Ghat', torch.ones(self.B, self.J))
        self.register_buffer('Ahat', torch.zeros(self.B, self.J))

        # Deadbands for param updates
        self.th_tau = 1e-3
        self.th_q   = 5e-5

        # Limits (scalars; broadcastable)
        self.torque_limit       = float(torque_limit)
        self.torque_rate_limit  = float(torque_rate_limit)

        # Physical parameters (buffers): allow scalar/(J,)/(B,J)
        self.register_buffer('I', torch.ones(self.B, self.J))
        self.register_buffer('m', torch.ones(self.B, self.J))
        self.register_buffer('g', torch.full((self.B, self.J), 9.81))
        self.register_buffer('l', torch.ones(self.B, self.J))

        # Blocks
        self.cmd  = CmdFilter2nd(n_envs, n_joints, Ts, fc_hz)
        self.diff = LevantOrder1(n_envs, n_joints, Ts, fd_hz)

    # ---------- Helpers ----------
    def set_limits(self, torque_limit: float = 100.0, torque_rate_limit: float = 2000.0):
        self.torque_limit = float(torque_limit)
        self.torque_rate_limit = float(torque_rate_limit)

    def set_critically_damped(self, f_hz):
        """
        Set Kp, Kd for ζ=1 (critical damping).
        f_hz: float or tensor (B,J) or (J,) of target natural frequency in Hz.
        """
        if not torch.is_tensor(f_hz):
            f_hz = torch.full((self.B, self.J), float(f_hz), device=self.Kp.device, dtype=self.Kp.dtype)
        elif f_hz.ndim == 1:  # (J,) -> (B,J)
            f_hz = f_hz.view(1, -1).expand(self.B, self.J).to(self.Kp.device, self.Kp.dtype)
        else:
            f_hz = f_hz.to(self.Kp.device, self.Kp.dtype)
        w = 2.0 * torch.pi * f_hz
        self.Kp.copy_(w * w)      # Kp = ω_n^2
        self.Kd.copy_(2.0 * w)    # Kd = 2 ω_n

    def set_gains(self, kp: float | torch.Tensor, kd: float | torch.Tensor):
        """Set Kp, Kd (scalar, (J,), or (B,J))."""
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

    def set_params(self,
                   I: float | torch.Tensor,
                   m: float | torch.Tensor,
                   g: float | torch.Tensor,
                   l: float | torch.Tensor):
        """Set (I,m,g,l). Accepts scalar, (J,), or (B,J) for each."""
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

    def reset(self, mask: Optional[torch.Tensor] = None):
        """
        Reset internal states.
        mask: (B,) boolean tensor, if given only reset selected environments; else reset all.
        """
        if mask is None:
            self.tau_prev.zero_()
            self.q_prev.zero_()
            self.qd_prev.zero_()
            self.qdd_prev.zero_()
            self.Ghat.fill_(1.0)
            self.Ahat.zero_()
            self.cmd.x0.zero_(); self.cmd.x1.zero_(); self.cmd.x2.zero_()
            self.diff.z0.zero_(); self.diff.z1.zero_()
        else:
            mask = mask.view(-1, 1).expand(-1, self.J)
            self.tau_prev[mask] = 0.0
            self.q_prev[mask]   = 0.0
            self.qd_prev[mask]  = 0.0
            self.qdd_prev[mask] = 0.0
            self.Ghat[mask]     = 1.0
            self.Ahat[mask]     = 0.0
            self.cmd.x0[mask]   = 0.0
            self.cmd.x1[mask]   = 0.0
            self.cmd.x2[mask]   = 0.0
            self.diff.z0[mask]  = 0.0
            self.diff.z1[mask]  = 0.0

    # ---------- Control tick ----------
    @torch.no_grad()
    def step(self, q_ref: torch.Tensor, q_meas: torch.Tensor, qd_meas: torch.Tensor, qdd_meas: torch.Tensor):
        """
        One control tick.
        Args:
            q_ref, q_meas, qd_meas: (B,J) tensors
        Returns:
            tau, tau_pd, tau_ff  -- tau already rate/limit-clamped
            - tau_pd: torque *increment* from PD (pre-scaling/clamp)
            - tau_ff: feedforward torque (pre-scaling/clamp), i.e., tau_prev + delta_tau_ff
        """
        # 1) Command filter builds (q_r, qd_r, qdd_r) from q_ref only
        # q_r, qd_r, qdd_r = self.cmd(q_ref)
        q_r = q_ref
        qd_r = torch.zeros_like(q_r)
        qdd_r = torch.zeros_like(q_r)

        # 2) Acceleration estimate from measured velocity
        qdd_hat = qdd_meas

        # 3) PD acceleration term  a_pd = Kd*(qd_r-qd) + Kp*(q_r-q)
        pd_term = self.Kd * (qd_r - qd_meas) + self.Kp * (q_r - q_meas)

        # 4) Choose Jacobians and optional A*Δq subtraction
        if self.use_model_jacobians:
            # Model Ginv = I (since G = 1/I)
            Ginv = torch.clamp(self.I, min=1e-9)
            # Model A = -(m g l / I) * cos(q_prev)
            A_model = -(self.m * self.g * self.l / torch.clamp(self.I, min=1e-9)) * torch.cos(self.q_prev)
            if self.use_AB:
                dq = q_meas - self.q_prev
                pd_term = pd_term - A_model * dq
        else:
            # Learned fallback (kept for completeness)
            Gabs = torch.clamp(self.Ghat.abs(), min=1e-3, max=1e3)
            Ginv = 1.0 / Gabs
            if self.use_AB:
                dq = q_meas - self.q_prev
                pd_term = pd_term - self.Ahat * dq

        # 5) Split INDI increments in acceleration domain
        #    ff_inc = qdd_r - qdd_prev  (use PREVIOUS accel for INDI)
        #    pd_inc = pd_term (already contains -A*Δq if enabled)
        ff_inc = qdd_r - qdd_hat
        pd_inc = pd_term

        # ---- Safety: cap acceleration increments per tick (tune as needed) ----
        a_ff_cap = 1500.0  # rad/s^2 per tick for feedforward
        a_pd_cap = 1000.0  # rad/s^2 per tick for PD
        ff_inc = torch.clamp(ff_inc, -a_ff_cap, a_ff_cap)
        pd_inc = torch.clamp(pd_inc, -a_pd_cap, a_pd_cap)

        # 6) Map to torque space and form desired torque before limits
        delta_tau_ff = Ginv * ff_inc
        delta_tau_pd = Ginv * pd_inc
        delta_tau_raw = delta_tau_ff + delta_tau_pd

        tau_ff_raw = self.tau_prev + delta_tau_ff    # keep for output (pre-limit FF)
        tau_pd     = delta_tau_pd                    # keep for output (pre-limit PD increment)
        tau_desired = self.tau_prev + delta_tau_raw

        # 7) Apply rate limit first
        max_step = self.torque_rate_limit * self.Ts
        tau_rate = torch.clamp(tau_desired,
                            self.tau_prev - max_step,
                            self.tau_prev + max_step)

        # 8) Available-margin scaling (soft anti-windup near ±limit)
        margin = (self.torque_limit - self.tau_prev.abs()).clamp_min(1e-6)    # Nm
        step_needed = (tau_rate - self.tau_prev).abs().clamp_min(1e-9)        # Nm
        alpha = (margin / step_needed).clamp(max=1.0)                         # 0..1, per joint
        tau_soft = self.tau_prev + alpha * (tau_rate - self.tau_prev)

        # 9) Hard clamp as last safety
        tau = torch.clamp(tau_soft, -self.torque_limit, self.torque_limit)

        # 10) Shift histories
        self.tau_prev.copy_(tau)
        self.q_prev.copy_(q_meas)
        self.qd_prev.copy_(qd_meas)
        self.qdd_prev.copy_(qdd_hat)

        return tau, tau_pd, tau_ff_raw

class CustomJointPositionAction(joint_actions.JointPositionAction):
    def __init__(self, cfg, env):
        # initialize the action term
        super().__init__(cfg, env)
        
        self.filtered_actions = torch.zeros_like(self.processed_actions)
        N, Ts = 10, 0.005
        self.ctrl = INDI_BatchedPendulum(
            n_envs=self.num_envs, n_joints=self.action_dim, Ts=Ts,
            joint_bw_hz=15.0, fc_hz=20.0, fd_hz=35.0,
            torque_limit=150.0, torque_rate_limit=2000.0,
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

    def apply_actions(self):
        # set position targets

        # get q_meas, qd_meas
        q_meas = self._asset.data.joint_pos
        qd_meas = self._asset.data.joint_vel
        qdd_meas = self._asset.data.joint_acc         
        # q_cmd = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        # q_cmd[:, 4] = 0.3; q_cmd[:, 6] = -0.6; q_cmd[:, 8] = 0.3 
        tau, _, _ = self.ctrl.step(self.processed_actions, q_meas, qd_meas, qdd_meas)

        # self.filtered_actions = 0.0 * self.filtered_actions + (1 - 0.0) * self.processed_actions
        # self._asset.set_joint_position_target(self.filtered_actions, joint_ids=self._joint_ids)
        self._asset.set_joint_effort_target(tau, joint_ids=self._joint_ids)

    def reset(self, env_ids) -> None:
        self._raw_actions[env_ids] = 0.0
        self.filtered_actions[env_ids] = 0.0
        self.ctrl.reset(env_ids)

@configclass
class Actions2PlayCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(class_type=CustomJointPositionAction, asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=False)

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
        joint_vel = ObservationTermCfg(func=mdp.joint_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=0.05)
        
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
        joint_cmd_pos = ObservationTermCfg(func=custom_mdp.joint_pos_and_cmd)
        joint_cmd_pos_error = ObservationTermCfg(func=custom_mdp.joint_vel_and_cmd_error)


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
        # self.physics_material.params["dynamic_friction_range"] = [0.1, 1.25]
        self.physics_material = None
        # self.add_base_mass.params["mass_distribution_params"] = [-1.0, 3.0]
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

        self.sim.render_interval = 8

        self.observations.policy.enable_corruption = False

        self.events.add_base_mass = None #type: ignore
        self.events.base_external_force_torque = None
        self.events.push_robot = None #type: ignore
        self.events.reset_base = None #type: ignore
        self.events.reset_robot_joints = None #type: ignore

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator.curriculum = False #type: ignore
        self.curriculum.terrain_levels = None #type: ignore

        self.commands.base_velocity.ranges.lin_vel_x = (0.7, 0.7)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.01, -0.01)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
   

        # self.viewer.asset_name = "robot"
        # self.viewer.origin_type = "asset_root"
        # self.viewer.eye = (0.0, 5.0, 2.0)


        # self.sim.use_fabric = False
        # self.sim.device = "cpu"
        # self.actions.joint_pos.scale = {"R_toe_joint": 0.001}
        # self.actions.joint_pos = Actions2PlayCfg()

 


    



    