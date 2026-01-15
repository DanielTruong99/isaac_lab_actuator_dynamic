from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

    from .commands import actions_cfg

class CustomJointPositionAction(JointPositionAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: CustomJointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: CustomJointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # self.coulomb_friction = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        # self.viscous_friction = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        # self.joint_pos_offset = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        # assume the joint ids are L -> R
        self.coulomb_friction = torch.tensor([
            0.623915, 1.844928, 4.754382, 8.390081, 0.393410, # left hip, hip2, thigh, calf, toe
            1.344627, 3.484132, 4.407304, 5.085046, 0.413043  # right hip, hip2, thigh, calf, toe
        ], device=self.device)
        self.coulomb_friction = self.coulomb_friction.repeat(self.num_envs, 1)

        self.viscous_friction = torch.tensor([
            0.179873, 0.593230, 0.097767, 3.834728, 0.023536, # left hip, hip2, thigh, calf, toe
            0.179873, 0.593230, 0.686212, 3.834728, 0.007213  # right hip, hip2, thigh, calf, toe
        ], device=self.device)
        self.viscous_friction = self.viscous_friction.repeat(self.num_envs, 1)

        # buffer
        self.filtered_actions = torch.zeros_like(self.raw_actions)
        self.prev_filtered_actions = torch.zeros_like(self.raw_actions)
        self.alpha = 0.6

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.filtered_actions[env_ids] = 0.0
        self._raw_actions[env_ids] = 0.0
        self.prev_filtered_actions[env_ids] = 0.0

    def set_joint_frictions(self, coulomb: torch.Tensor, viscous: torch.Tensor):
        """Set the joint friction coefficients.

        Args:
            coulomb (torch.Tensor): Coulomb friction coefficients of shape (num_envs, num_joints).
            viscous (torch.Tensor): Viscous friction coefficients of shape (num_envs, num_joints).
        """
        self.coulomb_friction = coulomb.to(dtype=torch.float32)
        self.viscous_friction = viscous.to(dtype=torch.float32)

    def set_joint_position_offset(self, offset: torch.Tensor):
        """Set the joint position offsets.

        Args:
            offset (torch.Tensor): Joint position offsets of shape (num_envs, num_joints).
        """
        self.joint_pos_offset = offset.to(dtype=torch.float32)

    # def process_actions(self, actions):
    #     super().process_actions(actions)

    # #     # alpha = 0.65  # fc = 3.5 hz 
    #     self.prev_filtered_actions = self.filtered_actions.clone()
    #     self.filtered_actions = self.alpha * self.filtered_actions + (1 - self.alpha) * self.processed_actions
        

    def apply_actions(self):
        # set position targets
        # alpha = 0.6442 # fc = 3.5 hz
        
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)

        # set friction torques
        dof_vel = self._asset.data.joint_vel[:, self._joint_ids]
        friction_torques = -self.coulomb_friction * torch.tanh(dof_vel/0.05) - self.viscous_friction * dof_vel
        self._asset.set_joint_effort_target(friction_torques, joint_ids=self._joint_ids)

@configclass
class CustomJointPositionActionCfg(JointActionCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = CustomJointPositionAction

    use_default_offset: bool = True
    """Whether to use default joint positions configured in the articulation asset as offset.
    Defaults to True.

    If True, this flag results in overwriting the values of :attr:`offset` to the default joint positions
    from the articulation asset.
    """