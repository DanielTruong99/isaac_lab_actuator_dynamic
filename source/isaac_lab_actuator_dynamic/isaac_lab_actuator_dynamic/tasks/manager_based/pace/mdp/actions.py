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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

    from . import actions_cfg

class CustomJointPositionAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: CustomJointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: CustomJointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        self.coulomb_friction = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)
        self.viscous_friction = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)
        self.joint_pos_offset = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)

        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

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

    def apply_actions(self):
        # set position targets
        joint_pos_cmds = self._env.command_manager._terms["motion"].joint_pos_cmds  # ensure motion command is updated
        self._asset.set_joint_position_target(joint_pos_cmds + self.joint_pos_offset, joint_ids=self._joint_ids)

        # set friction torques
        dof_vel = self._asset.data.joint_vel[:, self._joint_ids]
        friction_torques = -self.coulomb_friction * torch.tanh(dof_vel/0.05) - self.viscous_friction * dof_vel

        # get dof efforts
        # dof_efforts = self._env.command_manager._terms["motion"].joint_efforts
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