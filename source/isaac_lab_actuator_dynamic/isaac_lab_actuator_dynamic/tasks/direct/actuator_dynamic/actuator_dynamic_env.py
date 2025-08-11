# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_rotate, quat_apply

from .actuator_dynamic_env_cfg import ActuatorDynamic2EnvCfg
from .motions import MotionLoader


class ActuatorDynamic2Env(DirectRLEnv):
    cfg: ActuatorDynamic2EnvCfg

    def __init__(self, cfg: ActuatorDynamic2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.action_offset = 0.0
        self.action_scale = 1e-5

        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.previous_actions = torch.zeros(
            self.num_envs, self.cfg.action_space, device=self.device
        )

        # logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "is_terminated",
                "alive",
                "joint_pos_mimic",
                "joint_vel_mimic",
            ]
        }

        # load motion
        self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device) # type: ignore

        # DOF and key body indexes
        #! Need to be changed
        key_dof_names = ["L_hip_joint", "L_hip2_joint", "L_thigh_joint", "L_calf_joint", "L_toe_joint"]
        key_body_names = ["L_hip2", "L_thigh", "L_calf", "L_toe"]
        self.key_joint_indexes = self.robot.find_joints(key_dof_names)[0]
        self.ref_body_index = self.robot.data.body_names.index('base')
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        self.motion_dof_indexes = self._motion_loader.get_dof_index(key_dof_names)

    def _setup_scene(self):
        # add robot
        self.robot = Articulation(self.cfg.robot)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # add articulation to scene
        self.scene.articulations["robot"] = self.robot

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        """
            The action is the joint torques applied to the robot.
            The action is scaled to the joint torque limits and added to the joint efforts target.
        """
        target = self.action_offset + self.action_scale * self.actions
        # target = self.actions
        self.robot.set_joint_effort_target(target, self.key_joint_indexes)

    def _get_observations(self) -> dict:
        self.previous_actions = self.actions.clone()

        # build task observation
        obs = torch.cat(
            (
                self.robot.data.joint_pos[:, self.key_joint_indexes],
                self.robot.data.joint_vel[:, self.key_joint_indexes],
                # self.previous_actions,
            ),
            dim=-1,
        )

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # joint torques
        joint_torques = torch.sum(torch.square(self.robot.data.applied_torque[:, self.key_joint_indexes]), dim=1)

        # joint acceleration
        joint_accel = torch.sum(torch.square(self.robot.data.joint_acc[:, self.key_joint_indexes]), dim=1)

        # action rate
        action_rate = torch.sum(torch.square(self.actions - self.previous_actions), dim=1)

        # mimic recorded joint efforts
        applied_torques = self.robot.data.applied_torque[:, self.key_joint_indexes]

        # get motions
        times = self.episode_length_buf.cpu().numpy() * self._motion_loader.dt
        recorded_joint_pos, recorded_joint_vels, recorded_torques, = self._motion_loader.sample(num_samples=self.num_envs, times=times)
        joint_pos_error = torch.sum(torch.square(self.robot.data.joint_pos[:, self.key_joint_indexes] - recorded_joint_pos), dim=1)
        joint_pos_mimic = torch.exp(-joint_pos_error / 0.5)

        joint_vels_error = torch.sum(torch.square(self.robot.data.joint_vel[:, self.key_joint_indexes] - recorded_joint_vels), dim=1)
        joint_vel_mimic = torch.exp(-joint_vels_error / 0.5)

        rewards = {
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "is_terminated": self.reset_terminated.float() * self.cfg.terminated_scale * self.step_dt,
            "alive": (1.0 - self.reset_terminated.float()) * self.cfg.alive_scale * self.step_dt,
            "joint_pos_mimic": joint_pos_mimic * self.cfg.joint_pos_mimic_reward_scale * self.step_dt,
            "joint_vel_mimic": joint_vel_mimic * self.cfg.joint_vel_mimic_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        
        return reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            # died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
            # check dof position limits
            dof_pos = self.robot.data.joint_pos[:, self.key_joint_indexes]
            dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, self.key_joint_indexes, 0]
            dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, self.key_joint_indexes, 1]
            died = torch.any(
                torch.logical_or(
                    dof_pos < dof_lower_limits,
                    dof_pos > dof_upper_limits,
                ),
                dim=-1,
            )

            # check if the joint positions far from the recorded ones
            # times = self.episode_length_buf.cpu().numpy() * self._motion_loader.dt
            # recorded_joint_pos, _, _ = self._motion_loader.sample(num_samples=self.num_envs, times=times)
            # joint_pos_error = dof_pos - recorded_joint_pos
            # average_joint_pos_error = torch.mean(torch.abs(joint_pos_error), dim=1)
            # # if the average joint position error is greater than 20 degrees, then the agent is considered dead
            # died = torch.logical_or(died, average_joint_pos_error > np.deg2rad(20.0))   
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES # type: ignore
        self.robot.reset(env_ids) # type: ignore
        super()._reset_idx(env_ids) # type: ignore

        if self.cfg.reset_strategy == "default":
            joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy.startswith("random"):
            start = "start" in self.cfg.reset_strategy
            joint_pos, joint_vel = self._reset_strategy_random(env_ids, start)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, self.key_joint_indexes, env_ids) # type: ignore

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/terminated_far_from_reference"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

    # reset strategies

    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        return joint_pos, joint_vel

    def _reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # sample random motion times (or zeros if start is True)
        num_samples = env_ids.shape[0]
        times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples)
        # sample random motions
        (
            dof_positions,
            dof_velocities,
            dof_currents,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)

        # dof_positions = dof_positions.unsqueeze(-1)
        # dof_velocities = dof_velocities.unsqueeze(-1)

        # get DOFs state
        dof_pos = dof_positions[:, self.motion_dof_indexes]
        dof_vel = dof_velocities[:, self.motion_dof_indexes]

        return dof_pos, dof_vel

    # env methods

    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        # sample random motion times (or use the one specified)
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)
        times = (
            np.expand_dims(current_times, axis=-1)
            - self._motion_loader.dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()
        # get motions
        (
            dof_positions,
            dof_velocities,
            dof_currents,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)
        # compute AMP observation
        amp_observation = compute_obs(
            dof_positions,
            dof_velocities,
        )
        return amp_observation.view(-1, self.amp_observation_size)


@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)


@torch.jit.script
def compute_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
) -> torch.Tensor:
    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
        ),
        dim=-1,
    )
    return obs
