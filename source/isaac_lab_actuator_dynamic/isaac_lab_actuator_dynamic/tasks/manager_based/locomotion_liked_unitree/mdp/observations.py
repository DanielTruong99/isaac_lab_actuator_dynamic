import torch
import numpy as np

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets.articulation import Articulation

from ..walking_robot import WalkingRobotEnv

def get_phase(env: WalkingRobotEnv) -> torch.Tensor:
    sin_phase = torch.sin(2 * np.pi * env.phase ).unsqueeze(1)
    cos_phase = torch.cos(2 * np.pi * env.phase ).unsqueeze(1)
    return torch.cat([sin_phase, cos_phase], dim=-1)

def contact_state(env: WalkingRobotEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name] # type: ignore
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1.0 # type: ignore
    return is_contact.float()

def joint_torque(env: WalkingRobotEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids]

def joint_acc(env: WalkingRobotEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids]

def joint_pos_and_cmd(env: WalkingRobotEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    action = env.action_manager.action[:, asset_cfg.joint_ids]
    return torch.cat([asset.data.joint_pos[:, asset_cfg.joint_ids], action], dim=-1)

def joint_vel_and_cmd_error(env: WalkingRobotEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    action = env.action_manager.action[:, asset_cfg.joint_ids]
    error = action - asset.data.joint_pos[:, asset_cfg.joint_ids]
    return error