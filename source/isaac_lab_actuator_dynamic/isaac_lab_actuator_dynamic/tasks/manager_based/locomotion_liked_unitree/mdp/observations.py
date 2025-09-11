import torch
import numpy as np

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets.articulation import Articulation

from ..walking_robot import WalkingRobotEnv

def get_phase(env: WalkingRobotEnv) -> torch.Tensor:
    sin_phase = torch.sin(2 * np.pi * env.phase ).unsqueeze(1)
    cos_phase = torch.cos(2 * np.pi * env.phase ).unsqueeze(1)
    return torch.cat([sin_phase, cos_phase], dim=-1)

def joint_torque(env: WalkingRobotEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids]