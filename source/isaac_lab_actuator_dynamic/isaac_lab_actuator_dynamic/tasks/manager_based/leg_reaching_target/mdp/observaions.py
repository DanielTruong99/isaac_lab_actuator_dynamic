from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def body_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :].view(-1, 3) # type: ignore


def body_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.body_ang_vel_w[:, asset_cfg.body_ids, :].view(-1, 3) # type: ignore