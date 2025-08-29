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
from isaaclab.envs import DirectRLEnv, VecEnvStepReturn
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_rotate, quat_apply

from .actuator_dynamic_env_cfg import ActuatorDynamic2EnvCfg
from .motions import MotionLoader


class ActuatorDynamic2Env(DirectRLEnv):
    cfg: ActuatorDynamic2EnvCfg

    def __init__(self, cfg: ActuatorDynamic2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.action_offset = 0.0
        self.action_scale = 1.0

        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.joint_pos_cmds = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.previous_actions = torch.zeros(
            self.num_envs, self.cfg.action_space, device=self.device
        )
        self.residual_torques = torch.zeros_like(self.actions)

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
                "position_bonus",
            ]
        }

        # load motion and samples motion_ids
        self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device) # type: ignore
        motion_probs = torch.ones(self._motion_loader.num_motions, device=self.device) / self._motion_loader.num_motions
        self._motion_ids = torch.multinomial(motion_probs, self.num_envs, replacement=True)

        # DOF and key body indexes
        #! Need to be changed
        self.key_dof_names = ["L_hip_joint", "L_hip2_joint", "L_thigh_joint", "L_calf_joint", "L_toe_joint"]
        key_body_names = ["L_hip2", "L_thigh", "L_calf", "L_toe"]
        self.key_joint_indexes = self.robot.find_joints(self.key_dof_names)[0]
        self.ref_body_index = self.robot.data.body_names.index('base')
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        self.motion_dof_indexes = self._motion_loader.get_dof_index(self.key_dof_names)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Pre-process the actions before stepping through the physics.
        2. Apply the actions to the simulator and step through the physics in a decimated manner.
        3. Compute the reward and done signals.
        4. Reset environments that have terminated or reached the maximum episode length.
        5. Apply interval events if they are enabled.
        6. Compute observations.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        action = action.to(self.device)
        # add action noise
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)

        # process actions
        self._pre_physics_step(action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        self._post_physics_steps()

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()
            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        self.obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _post_physics_steps(self):
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)        
        
        # resample reference motions for used to compute observations, rewards, ...
        times = self.episode_length_buf * self._motion_loader.dt[self._motion_ids]
        self.recorded_joint_pos, self.recorded_joint_vels, self.recorded_torques = self._motion_loader.sample(self._motion_ids, motion_times=times)

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
        
        # get joint position cmds
        times = self.episode_length_buf * self._motion_loader.dt[self._motion_ids]
        self.joint_pos_cmds, _, _, = self._motion_loader.sample(self._motion_ids, times)

    def _apply_action(self):
        """
            The action is the joint torques applied to the robot.
            The action is scaled to the joint torque limits and added to the joint efforts target.
        """
        self.residual_torques = self.action_offset + self.action_scale * self.actions
        # target = self.actions
        self.robot.set_joint_effort_target(self.residual_torques, self.key_joint_indexes)

        # get joint positions cmds from motion loader to feed into this one
        self.robot.set_joint_position_target(self.joint_pos_cmds, self.key_joint_indexes)

    def _get_observations(self) -> dict:
        self.previous_actions = self.actions.clone()

        # build task observation
        obs = torch.cat(
            (
                self.joint_pos_cmds,
                self.robot.data.joint_pos[:, self.key_joint_indexes],
                self.robot.data.joint_vel[:, self.key_joint_indexes],
                self.previous_actions,
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

        # reference motions
        joint_pos_error = torch.sum(torch.square(self.robot.data.joint_pos[:, self.key_joint_indexes] - self.recorded_joint_pos), dim=1)
        # joint_pos_mimic = torch.exp(-joint_pos_error / 0.5)
        joint_pos_mimic = (
            - 15.0 * joint_pos_error
            + 1.0 / (torch.exp(-10 * joint_pos_error) + torch.exp(10 * joint_pos_error) + 1e-6)
            + 1.0 / (torch.exp(-700 * joint_pos_error) + torch.exp(700 * joint_pos_error) + 1e-6)
        )
        joint_pos_error_mean = torch.mean(torch.abs(self.robot.data.joint_pos[:, self.key_joint_indexes] - self.recorded_joint_pos), dim=1)
        position_bonus = joint_pos_error_mean < 1e-3
        position_bonus = position_bonus.float()

        joint_vels_error = torch.sum(torch.square(self.robot.data.joint_vel[:, self.key_joint_indexes] - self.recorded_joint_vels), dim=1)
        joint_vel_mimic = torch.exp(-joint_vels_error / 0.5)

        rewards = {
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "is_terminated": self.reset_terminated.float() * self.cfg.terminated_scale * self.step_dt,
            "alive": (1.0 - self.reset_terminated.float()) * self.cfg.alive_scale * self.step_dt,
            "joint_pos_mimic": joint_pos_mimic * self.cfg.joint_pos_mimic_reward_scale * self.step_dt,
            "joint_vel_mimic": joint_vel_mimic * self.cfg.joint_vel_mimic_reward_scale * self.step_dt,
            "position_bonus": position_bonus * self.cfg.joint_pos_bonus_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        
        return reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1 
        # time out by end of reference motions
        current_times = self.episode_length_buf * self._motion_loader.dt[self._motion_ids]
        time_out = torch.logical_or(time_out, current_times >= self._motion_loader.duration[self._motion_ids])
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

        # add noise
        if self.cfg.randomize_initial_state:
            if self.cfg.random_scale_cfg.joint_pos:
                for joint_name, noise_range in self.cfg.random_scale_cfg.joint_pos.items():
                    joint_index = self.key_dof_names.index(joint_name)
                    a, b = noise_range
                    noise = (b - a) * torch.rand(len(env_ids), device=self.device) + a
                    joint_pos[:, joint_index] += noise
            if self.cfg.random_scale_cfg.joint_vel:
                for joint_name, noise_range in self.cfg.random_scale_cfg.joint_vel.items():
                    joint_index = self.key_dof_names.index(joint_name)
                    a, b = noise_range
                    noise = (b - a) * torch.rand(len(env_ids), device=self.device) + a
                    joint_vel[:, joint_index] += noise

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
        times = self._motion_loader.sample_times(self._motion_ids)
        if start:
            times = torch.zeros_like(times)

        # sample random motions
        (
            dof_positions,
            dof_velocities,
            dof_currents,
        ) = self._motion_loader.sample(self._motion_ids, motion_times=times)

        # get DOFs state
        dof_pos = dof_positions[env_ids][:, self.motion_dof_indexes]
        dof_vel = dof_velocities[env_ids][:, self.motion_dof_indexes]

        return dof_pos, dof_vel



class ActuatorDynamic2PlayEnv(ActuatorDynamic2Env):
    pass

@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)

