from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_error_magnitude, yaw_quat

from isaac_lab_actuator_dynamic.tasks.manager_based.sim2real_dm_locomotion.dm_mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]

def track_lin_vel_xy_exp_max(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    
    tar_speed = torch.norm(env.command_manager.get_command(command_name)[:, 0:1], dim=1)
    tar_dir = env.command_manager.get_command(command_name)[:, 1:]
    tar_dir_norm = torch.norm(tar_dir, dim=-1, keepdim=True) + 1e-8
    orientation_vcmd_xy = tar_dir / tar_dir_norm
    projected_vxy = torch.sum(asset.data.root_lin_vel_w[:, :2] * orientation_vcmd_xy, dim=-1)
    lin_error = tar_speed - projected_vxy
    lin_error_max =  torch.square(lin_error)

    lin_error_d = asset.data.root_lin_vel_w[:, :2] - projected_vxy.unsqueeze(-1) * orientation_vcmd_xy
    lin_error_max_d =  torch.sum(torch.square(lin_error_d), dim=-1)

    quat_yaw = yaw_quat(asset.data.root_quat_w)
    x_axis = torch.zeros_like(asset.data.root_pos_w)
    x_axis[:, 0] = 1.0
    heading_dir = quat_apply(quat_yaw, x_axis)
    heading_dir_xy = heading_dir[:, :2] / (torch.norm(heading_dir[:, :2], dim=-1, keepdim=True) + 1e-8)
    heading_proj = torch.sum(orientation_vcmd_xy * heading_dir_xy, dim=-1)
    heading_error = torch.square(heading_proj)
    heading_reward = torch.clamp_min(heading_error, 0.0)

    return torch.exp(lin_error_max * (-2.0) + lin_error_max_d * (-0.2)) + heading_reward

def track_lin_vel_xy_exp_d(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    
    norm_vcmd_xy = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    orientation_vcmd_xy = env.command_manager.get_command(command_name)[:, :2] / (norm_vcmd_xy.unsqueeze(-1) + 1e-8)
    projected_vxy = torch.sum(asset.data.root_lin_vel_b[:, :2] * orientation_vcmd_xy, dim=-1)
    lin_error = asset.data.root_lin_vel_b[:, :2] - projected_vxy.unsqueeze(-1) * orientation_vcmd_xy
    lin_error_max =  torch.sum(torch.square(lin_error), dim=-1)
    return torch.exp(lin_error_max * (-0.2))

def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    vel_command = env.command_manager.get_term("base_velocity")

    is_standing = (~vel_command.is_standing_env).float()
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    
    return torch.exp(-error.mean(-1) / std**2) * is_standing


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )

    vel_command = env.command_manager.get_term("base_velocity")
    is_standing = (~vel_command.is_standing_env).float()
    return torch.exp(-error.mean(-1) / std**2) * is_standing

def motion_joint_pos_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error =  torch.square(command.joint_pos - command.robot_joint_pos)
    
    return torch.exp(-error.mean(-1) / std**2)

def motion_joint_vel_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error =  torch.square(command.joint_vel - command.robot_joint_vel)
    
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward