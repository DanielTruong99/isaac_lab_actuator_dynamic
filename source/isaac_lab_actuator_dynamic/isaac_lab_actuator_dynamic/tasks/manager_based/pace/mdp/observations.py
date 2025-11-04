from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv



def joint_pos_sim_real(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    motion_commander = env.command_manager.get_term("motion")

    #! the joint indices in the motion command correspond to the ones in the asset
    joint_pos_sim = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_pos_real = motion_commander.joint_pos[:, asset_cfg.joint_ids]

    return torch.cat([joint_pos_sim, joint_pos_real], dim=-1)

def joint_effort_sim_real(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint efforts of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their efforts  returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    motion_commander = env.command_manager.get_term("motion")

    #! the joint indices in the motion command correspond to the ones in the asset
    joint_effort_sim = asset.data.applied_torque[:, asset_cfg.joint_ids]
    joint_effort_real = motion_commander.joint_efforts[:, asset_cfg.joint_ids]

    return torch.cat([joint_effort_sim, joint_effort_real], dim=-1)