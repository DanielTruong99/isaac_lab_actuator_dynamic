from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .commands import MotionCommand



def motion_position_error(
    env: ManagerBasedRLEnv, command_name: str, 
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene["robot"]
    error = command.joint_pos - robot.data.joint_pos[:, command._joint_ids]
    # coeff = torch.tensor([1.0, 5.0, 10.0, 5.0, 1.0]).to(error.device)
    # error = error * coeff
    error_l2= torch.sum(
        torch.square(error), dim=-1
    )
    return error_l2

def motion_velocity_error(
    env: ManagerBasedRLEnv, command_name: str, 
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene["robot"]
    error = command.joint_vels - robot.data.joint_vel[:, command._joint_ids]
    # coeff = torch.tensor([1.0, 5.0, 10.0, 5.0, 1.0]).to(error.device)
    # error = error * coeff
    error_l2= torch.sum(
        torch.square(error), dim=-1
    )
    return error_l2