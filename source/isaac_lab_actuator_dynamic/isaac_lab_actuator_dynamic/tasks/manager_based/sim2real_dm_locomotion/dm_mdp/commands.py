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


class MotionLoader:
    def __init__(self, motion_files: str, num_envs, body_indexes: Sequence[int], device: str = "cpu"):
        joint_pos_list = []
        joint_vel_list = []
        body_pos_w_list = []
        body_quat_w_list = []
        body_lin_vel_w_list = []
        body_ang_vel_w_list = []
        time_step_total_list = []
        for index, motion_file in enumerate(motion_files):
            assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
            data = np.load(motion_file)

            # Assuming the expected joint names [L_hip_joint -> L_toe_joint, R_hip_joint -> R_toe_joint]
            # Assuming the expected body names [L_hip -> L_toe, R_hip -> R_toe]
            joint_pos = data["dof_positions"]
            joint_pos_list.append(joint_pos)
            time_step_total_list.append(joint_pos.shape[0])
            

            joint_vel = data["dof_vels"]
            joint_vel_list.append(joint_vel)

            body_pos_w = data["body_positions_w"]
            # Adjust height
            min_height = np.min(body_pos_w[:, :, 2])
            body_pos_w[:, :, 2] -= min_height 
            body_pos_w_list.append(body_pos_w)

            body_quat_w = data["body_quats_w"]
            body_quat_w_list.append(body_quat_w)

            body_lin_vel_w =  data["body_lin_vels_w"]
            body_lin_vel_w_list.append(body_lin_vel_w)

            body_ang_vel_w = data["body_ang_vels_w"]
            body_ang_vel_w_list.append(body_ang_vel_w)
        
        self.time_step_total_all = torch.tensor(time_step_total_list, dtype=torch.long, device=device)
        self.joint_pos = torch.tensor(np.concatenate(joint_pos_list, axis=0), dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(np.concatenate(joint_vel_list, axis=0), dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(np.concatenate(body_pos_w_list, axis=0), dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(np.concatenate(body_quat_w_list, axis=0), dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(np.concatenate(body_lin_vel_w_list, axis=0), dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(np.concatenate(body_ang_vel_w_list, axis=0), dtype=torch.float32, device=device)

        self._body_indexes = body_indexes
        self.fps = data["fps"]

        self.num_envs = num_envs
        self.motion_files = motion_files
        self.motion_ids = torch.multinomial(input=torch.tensor([0.3, 0.3, 0.1, 0.1, 0.1, 0.1], device=device), num_samples=num_envs, replacement=True)
        self.time_step_cumsum = torch.cumsum(self.time_step_total_all, dim=0)
        self.end_indices = self.time_step_cumsum[self.motion_ids]

        time_step_rolled = self.time_step_total_all.roll(shifts=1)
        time_step_rolled[0] = 0
        self.cumsum_time_step_rolled = torch.cumsum(time_step_rolled, dim=0)
        self.start_indices = self.cumsum_time_step_rolled[self.motion_ids] 
        self.time_step_total = self.time_step_total_all[self.motion_ids]
    
    def assign_motion_ids(self, motion_ids: torch.Tensor, env_ids: Sequence[int]):
        self.motion_ids[env_ids] = motion_ids
        self.end_indices[env_ids] = self.time_step_cumsum[motion_ids]
        self.start_indices[env_ids] = self.cumsum_time_step_rolled[motion_ids]
        self.time_step_total[env_ids] = self.time_step_total_all[motion_ids]

    def resample_motions(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        motion_ids = torch.multinomial(
            input=torch.tensor([0.3, 0.3, 0.1, 0.1, 0.1, 0.1], device=self.motion_ids.device),
            num_samples=len(env_ids),
            replacement=True,
        )
        self.end_indices[env_ids] = self.time_step_cumsum[motion_ids]
        self.start_indices[env_ids] = self.cumsum_time_step_rolled[motion_ids]
        self.time_step_total[env_ids] = self.time_step_total_all[motion_ids]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MotionLoader(self.cfg.motion_file, self.num_envs, self.body_indexes, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.motion.start_indices + self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.motion.start_indices + self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.motion.start_indices + self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.motion.start_indices + self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.motion.start_indices + self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.motion.start_indices + self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.motion.start_indices + self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.motion.start_indices + self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.motion.start_indices + self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.motion.start_indices + self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )
        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _uniform_sampling(self, env_ids: Sequence[int]):
        # Ignore failure statistics — always uniform sampling
        num_envs = len(env_ids)

        # Compute sampled timesteps
        self.time_steps[env_ids] = (
            torch.rand(num_envs, device=self.device)
            * (self.motion.time_step_total[env_ids] - 1)
        ).long()

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        # self.motion.resample_motions(env_ids)
        self._uniform_sampling(env_ids)
        

        # root_pos = self.body_pos_w[:, 0].clone()
        # root_ori = self.body_quat_w[:, 0].clone()
        # root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        # root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        # range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        # ranges = torch.tensor(range_list, device=self.device)
        # rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        # root_pos[env_ids] += rand_samples[:, 0:3]
        # orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        # root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        # range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        # ranges = torch.tensor(range_list, device=self.device)
        # rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        # root_lin_vel[env_ids] += rand_samples[:, :3]
        # root_ang_vel[env_ids] += rand_samples[:, 3:]

        # joint_pos = self.joint_pos.clone()
        # joint_vel = self.joint_vel.clone()

        # joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        # soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        # joint_pos[env_ids] = torch.clip(
        #     joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        # )
        # self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        # self.robot.write_root_state_to_sim(
        #     torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
        #     env_ids=env_ids,
        # )

    def _update_command(self):
        self.time_steps += 1
        times = self.time_steps + self.motion.start_indices
        env_ids = torch.where(times >= self.motion.end_indices)[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)


    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)