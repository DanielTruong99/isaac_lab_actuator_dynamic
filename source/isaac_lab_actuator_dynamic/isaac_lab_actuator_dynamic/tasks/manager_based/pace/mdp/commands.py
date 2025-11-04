from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaac_lab_actuator_dynamic.tasks.manager_based.pace.motions import MotionLoader



class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg
    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device) # type: ignore
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._joint_ids, _ = self.robot.find_joints(name_keys=self._motion_loader.dof_names)

        motion_probs = torch.ones(self._motion_loader.num_motions, device=self.device) / self._motion_loader.num_motions
        self._motion_ids = torch.multinomial(motion_probs, self.num_envs, replacement=True)

        # prepare buffers
        self._joint_pos = torch.zeros((self.num_envs, len(self._joint_ids)), device=self.device)
        self._joint_vels = torch.zeros((self.num_envs, len(self._joint_ids)), device=self.device)
        self._joint_pos_cmds = torch.zeros((self.num_envs, len(self._joint_ids)), device=self.device)
        self._joint_efforts = torch.zeros((self.num_envs, len(self._joint_ids)), device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.joint_pos
    
    @property
    def joint_pos(self) -> torch.Tensor:
        return self._joint_pos
    
    @property
    def joint_pos_cmds(self) -> torch.Tensor:
        return self._joint_pos_cmds

    @property
    def joint_vels(self) -> torch.Tensor:
        return self._joint_vels
    
    @property
    def joint_efforts(self) -> torch.Tensor:
        return self._joint_efforts

    def reset_envs(self, env_ids: Sequence[int]):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        times = self.time_steps * self._env.step_dt
        (
            self._joint_pos, 
            self._joint_vels, 
            self._joint_pos_cmds, 
            self._joint_efforts
        ) = self._motion_loader.sample(self._motion_ids, motion_times=times)

        joint_pos = self.joint_pos.clone()
        joint_vels = self.joint_vels.clone()
        
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vels[env_ids], env_ids=env_ids, joint_ids=self._joint_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        self.time_steps[env_ids] = 0

        times = self.time_steps * self._env.step_dt
        (
            self._joint_pos, 
            self._joint_vels, 
            self._joint_pos_cmds, 
            self._joint_efforts
        ) = self._motion_loader.sample(self._motion_ids, motion_times=times)

        joint_pos = self.joint_pos.clone()
        joint_vels = self.joint_vels.clone()
        
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vels[env_ids], env_ids=env_ids, joint_ids=self._joint_ids)

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps * self._env.step_dt >= self._motion_loader.duration[self._motion_ids])[0]
        self._resample_command(env_ids)

        # the times should be env step
        times = self.time_steps * self._env.step_dt
        (
            self._joint_pos, 
            self._joint_vels, 
            self._joint_pos_cmds, 
            self._joint_efforts
        ) = self._motion_loader.sample(self._motion_ids, motion_times=times)

    def _update_metrics(self):
        pass

@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: list[str] = MISSING
