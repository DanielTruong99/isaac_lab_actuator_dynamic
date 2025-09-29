from __future__ import annotations

import numpy as np
from torch import distributions
from typing import TYPE_CHECKING, Optional

import torch
import isaaclab.envs.mdp as mdp
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg

from ..walking_robot import WalkingRobotEnv

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

def weighted_joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    original_reward = mdp.joint_torques_l2(env, asset_cfg)
    v_cmd = env.command_manager.get_command("base_velocity")
    norm_vxy_cmd = torch.norm(v_cmd[:, 0:2], dim=-1)
    result = original_reward * (torch.exp(-1.5 * norm_vxy_cmd))
    return result



def weighted_track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Computes a weighted tracking reward for linear velocity in the XY plane using an exponential function.
    This function calculates the original reward using the `track_lin_vel_xy_exp` method from the `mdp` module.
    It then adjusts this reward based on the norm of the commanded base velocity, applying an exponential decay.
    Args:
        env (ManagerBasedRLEnv): The environment instance containing the command manager and other necessary components.
        std (float): The standard deviation used in the original reward calculation.
        command_name (str): The name of the command to be tracked.
        asset_cfg (SceneEntityCfg, optional): Configuration for the scene entity, defaulting to a robot configuration.
    Returns:
        torch.Tensor: The computed weighted reward as a tensor.
    """

    original_reward = mdp.track_lin_vel_xy_exp(env, std, command_name, asset_cfg)
    v_cmd = env.command_manager.get_command("base_velocity")
    norm_vxy_cmd = torch.norm(v_cmd[:, 0:2], dim=-1)
    result = original_reward * (1.0 - torch.exp(-1.5 * norm_vxy_cmd))
    return result


def weighted_track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Computes a weighted reward based on the angular velocity around the z-axis and the commanded base velocity.
    This function first calculates the original reward using the `track_ang_vel_z_exp` method from the `mdp` module.
    It then adjusts this reward by a factor that depends on the norm of the commanded base velocity in the x and y directions.
    Args:
        env (ManagerBasedRLEnv): The environment instance containing the state and command managers.
        std (float): The standard deviation used in the original reward calculation.
        command_name (str): The name of the command to track.
        asset_cfg (SceneEntityCfg, optional): Configuration for the scene entity, default is a robot.
    Returns:
        torch.Tensor: The computed weighted reward.
    """

    original_reward = mdp.track_ang_vel_z_exp(env, std, command_name, asset_cfg)
    v_cmd = env.command_manager.get_command("base_velocity")
    norm_vxy_cmd = torch.norm(v_cmd[:, 0:2], dim=-1)
    result = original_reward * (1.0 - torch.exp(-1.5 * norm_vxy_cmd))
    return result


def weighted_is_alive(env: WalkingRobotEnv) -> torch.Tensor:
    """
    Calculate a weighted "is alive" reward for the walking robot environment.

    This function computes the original "is alive" reward from the MDP and 
    applies a weighting factor based on the norm of the commanded base velocity 
    in the x and y directions. The weighting factor is an exponential decay 
    function that reduces the reward as the commanded velocity increases.

    Args:
        env (WalkingRobotEnv): The walking robot environment instance.

    Returns:
        torch.Tensor: The weighted "is alive" reward.
    """

    original_reward = mdp.is_alive(env)
    v_cmd = env.command_manager.get_command("base_velocity")
    norm_vxy_cmd = torch.norm(v_cmd[:, 0:2], dim=-1)
    result = original_reward * torch.exp(-1.5 * norm_vxy_cmd)
    return result


def weighted_feet_schedule_contact(env: WalkingRobotEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Calculate the weighted feet schedule contact reward for a walking robot environment.

    This function computes the original reward using the feet_schedule_contact function and then
    applies a weighting based on the commanded base velocity. The weighting is an exponential
    decay function of the norm of the commanded velocity in the x and y directions.

    Args:
        env (WalkingRobotEnv): The walking robot environment instance.
        sensor_cfg (SceneEntityCfg): The configuration for the scene entity sensors.

    Returns:
        torch.Tensor: The weighted reward tensor.
    """

    original_reward = feet_schedule_contact(env, sensor_cfg)
    v_cmd = env.command_manager.get_command("base_velocity")
    norm_vxy_cmd = torch.norm(v_cmd[:, 0:2], dim=-1)
    result = original_reward * (torch.exp(-1.5 * norm_vxy_cmd))
    return result

def feet_schedule_contact(env: WalkingRobotEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Computes the contact schedule for the feet of a walking robot.
    scheduler | is_contact | result
    --------------------------------
    0         | 0          | 1
    0         | 1          | 0
    1         | 0          | 0
    1         | 1          | 1


    Args:
        env (WalkingRobotEnv): The environment containing the walking robot.
        sensor_cfg (SceneEntityCfg): Configuration for the contact sensor.

    Returns:
        torch.Tensor: A tensor indicating the contact schedule for each environment.
    """

    result = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32, requires_grad=False)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name] # type: ignore
    net_contact_forces = contact_sensor.data.net_forces_w_history

    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1.0 # type: ignore
    result = result + ~(is_contact[:, 0] ^ (env.phase_left < 0.55)) + ~(is_contact[:, 1] ^ (env.phase_right < 0.55))
    return result

def feet_schedule_contact_with_cmd(env: WalkingRobotEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Computes the contact schedule for the feet of a walking robot.
    norm > 0.1 |scheduler | is_contact | result
    --------------------------------
    1          | 0        | 0          | 1
    1          | 0        | 1          | 0
    1          | 1        | 0          | 0
    1          | 1        | 1          | 1
    0          | 0        | 0          | 0
    0          | 0        | 1          | 1
    0          | 1        | 0          | 0
    0          | 1        | 1          | 1


    Args:
        env (WalkingRobotEnv): The environment containing the walking robot.
        sensor_cfg (SceneEntityCfg): Configuration for the contact sensor.

    Returns:
        torch.Tensor: A tensor indicating the contact schedule for each environment.
    """
    v_cmd = env.command_manager.get_command("base_velocity")
    is_vcmd_gt = torch.norm(v_cmd, dim=-1) > 0.1

    result = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32, requires_grad=False)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name] # type: ignore
    net_contact_forces = contact_sensor.data.net_forces_w_history

    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1.0 # type: ignore

    # left_leg_result = (~(is_contact[:, 0] ^ (env.phase_left < 0.55))) * is_vcmd_gt + is_contact[:, 0] * (~is_vcmd_gt)
    # right_leg_result = (~(is_contact[:, 1] ^ (env.phase_right < 0.55))) * is_vcmd_gt + is_contact[:, 1] * (~is_vcmd_gt)
    left_leg_result = (~(is_contact[:, 0] ^ (env.phase_left < 0.55))) * is_vcmd_gt
    right_leg_result = (~(is_contact[:, 1] ^ (env.phase_right < 0.55))) * is_vcmd_gt
    result = result + left_leg_result + right_leg_result
    return result

def stand_still_contact(env: WalkingRobotEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Calculate the stand still contact reward for a walking robot environment.

    This function computes a reward based on whether the feet are in contact with the ground
    when the commanded base velocity is below a certain threshold. If the commanded velocity
    is low, the reward is given for having contact; otherwise, no reward is given.

    Args:
        env (WalkingRobotEnv): The walking robot environment instance.
        sensor_cfg (SceneEntityCfg): The configuration for the contact sensor.

    Returns:
        torch.Tensor: A tensor indicating the stand still contact reward for each environment.
    """

    result = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32, requires_grad=False)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name] # type: ignore
    net_contact_forces = contact_sensor.data.net_forces_w_history

    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1.0 # type: ignore
    v_cmd = env.command_manager.get_command("base_velocity")
    is_vcmd_lt = torch.norm(v_cmd,dim=-1) < 0.1

    result = result + (~is_contact[:, 0] * is_vcmd_lt) + (~is_contact[:, 1] * is_vcmd_lt)
    return result

def stand_still(env: WalkingRobotEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Calculate the stand still contact reward for a walking robot environment.

    This function computes a reward based on whether the feet are in contact with the ground
    when the commanded base velocity is below a certain threshold. If the commanded velocity
    is low, the reward is given for having contact; otherwise, no reward is given.

    Args:
        env (WalkingRobotEnv): The walking robot environment instance.
        sensor_cfg (SceneEntityCfg): The configuration for the contact sensor.

    Returns:
        torch.Tensor: A tensor indicating the stand still contact reward for each environment.
    """
    v_cmd = env.command_manager.get_command("base_velocity")
    is_vcmd_lt = torch.norm(v_cmd,dim=-1) < 0.1

    asset = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1) * is_vcmd_lt

def feet_height(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Calculate the height of the feet relative to the base position of the robot and return a reward based on the 
    feet height when there is no contact.

    Args:
        env: The environment object containing the scene and sensors.
        sensor_cfg (SceneEntityCfg): Configuration for the contact sensor.
        asset_cfg (SceneEntityCfg, optional): Configuration for the robot asset. Defaults to SceneEntityCfg("robot").

    Returns:
        torch.Tensor: A tensor containing the calculated reward based on the feet height.
    """
    
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0 # type: ignore
    asset = env.scene[asset_cfg.name]

    v_cmd = env.command_manager.get_command("base_velocity")
    is_vcmd_gt = torch.norm(v_cmd, dim=-1) > 0.1

    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    base_pos_w = asset.data.root_pos_w
    feet_height = feet_pos_w - base_pos_w.unsqueeze(1) 

    result = (~is_contact) * torch.square(feet_height[:, :, 2] - (-0.7405))
    return torch.sum(result, dim=1) * is_vcmd_gt


def action_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        env.action_manager.action[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        env.action_manager.action[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)

def feet_distance(env: ManagerBasedRLEnv,
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                  feet_links_name: list[str]=["foot_[RL]_Link"],
                  min_feet_distance: float = 0.1,
                  max_feet_distance: float = 1.0,)-> torch.Tensor:
    # Penalize base height away from target
    asset = env.scene[asset_cfg.name]
    feet_links_idx = asset.find_bodies(feet_links_name)[0]
    feet_pos = asset.data.body_link_pos_w[:,feet_links_idx]
    # feet distance on x-y plane
    feet_distance = torch.norm(feet_pos[:, 0, :2] - feet_pos[:, 1, :2], dim=-1)
    reward = torch.clip(min_feet_distance - feet_distance, 0, 1)
    reward += torch.clip(feet_distance - max_feet_distance, 0, 1)
    return reward

def feet_regulation(env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    foot_radius: float,
    base_height_target: float,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    feet_height = torch.clip(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - foot_radius, 0, 1
    )  # TODO: change to the height relative to the vertical projection of the terrain
    feet_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]

    height_scale = torch.exp(-feet_height / base_height_target)
    reward = torch.sum(height_scale * torch.square(torch.norm(feet_vel_xy, dim=-1)), dim=1)
    return reward

def foot_landing_vel(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        foot_radius: float,
        about_landing_threshold: float,
) -> torch.Tensor:
    """Penalize high foot landing velocities"""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    z_vels = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 0.1

    foot_heights = torch.clip(
    asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - foot_radius, 0, 1
    )  # TODO: change to the height relative to the vertical projection of the terrain

    about_to_land = (foot_heights < about_landing_threshold) & (~contacts) & (z_vels < 0.0)
    landing_z_vels = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
    reward = torch.sum(torch.square(landing_z_vels), dim=1)
    return reward

def joint_powers_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint powers on the articulation using L1-kernel"""

    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(torch.mul(asset.data.applied_torque, asset.data.joint_vel)), dim=1)

class GaitReward(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)

        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]

        # extract the used quantities (to enable type-hinting)
        self.contact_sensor: ContactSensor = env.scene.sensors[self.sensor_cfg.name]
        self.asset = env.scene[self.asset_cfg.name]

        # Store configuration parameters
        self.force_scale = float(cfg.params["tracking_contacts_shaped_force"])
        self.vel_scale = float(cfg.params["tracking_contacts_shaped_vel"])
        self.force_sigma = cfg.params["gait_force_sigma"]
        self.vel_sigma = cfg.params["gait_vel_sigma"]
        self.kappa_gait_probs = cfg.params["kappa_gait_probs"]
        self.command_name = cfg.params["command_name"]
        self.dt = env.step_dt

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        tracking_contacts_shaped_force,
        tracking_contacts_shaped_vel,
        gait_force_sigma,
        gait_vel_sigma,
        kappa_gait_probs,
        command_name,
        sensor_cfg,
        asset_cfg,
    ) -> torch.Tensor:
        """Compute the reward.

        The reward combines force-based and velocity-based terms to encourage desired gait patterns.

        Args:
            env: The RL environment instance.

        Returns:
            The reward value.
        """

        gait_params = env.command_manager.get_command(self.command_name)

        # Update contact targets
        desired_contact_states = self.compute_contact_targets(gait_params)

        # Force-based reward
        foot_forces = torch.norm(self.contact_sensor.data.net_forces_w[:, self.sensor_cfg.body_ids], dim=-1)
        force_reward = self._compute_force_reward(foot_forces, desired_contact_states)

        # Velocity-based reward
        foot_velocities = torch.norm(self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids], dim=-1)
        velocity_reward = self._compute_velocity_reward(foot_velocities, desired_contact_states)

        # Combine rewards
        total_reward = force_reward + velocity_reward
        return total_reward

    def compute_contact_targets(self, gait_params):
        """Calculate desired contact states for the current timestep."""
        frequencies = gait_params[:, 0]
        offsets = gait_params[:, 1]
        durations = torch.cat(
            [
                gait_params[:, 2].view(self.num_envs, 1),
                gait_params[:, 2].view(self.num_envs, 1),
            ],
            dim=1,
        )

        assert torch.all(frequencies > 0), "Frequencies must be positive"
        assert torch.all((offsets >= 0) & (offsets <= 1)), "Offsets must be between 0 and 1"
        assert torch.all((durations > 0) & (durations < 1)), "Durations must be between 0 and 1"

        gait_indices = torch.remainder(self._env.episode_length_buf * self.dt * frequencies, 1.0)

        # Calculate foot indices
        foot_indices = torch.remainder(
            torch.cat(
                [gait_indices.view(self.num_envs, 1), (gait_indices + offsets + 1).view(self.num_envs, 1)],
                dim=1,
            ),
            1.0,
        )

        # Determine stance and swing phases
        stance_idxs = foot_indices < durations
        swing_idxs = foot_indices > durations

        # Adjust foot indices based on phase
        foot_indices[stance_idxs] = torch.remainder(foot_indices[stance_idxs], 1) * (0.5 / durations[stance_idxs])
        foot_indices[swing_idxs] = 0.5 + (torch.remainder(foot_indices[swing_idxs], 1) - durations[swing_idxs]) * (
            0.5 / (1 - durations[swing_idxs])
        )

        # Calculate desired contact states using von mises distribution
        smoothing_cdf_start = distributions.normal.Normal(0, self.kappa_gait_probs).cdf
        desired_contact_states = smoothing_cdf_start(foot_indices) * (
            1 - smoothing_cdf_start(foot_indices - 0.5)
        ) + smoothing_cdf_start(foot_indices - 1) * (1 - smoothing_cdf_start(foot_indices - 1.5))

        return desired_contact_states

    def _compute_force_reward(self, forces: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
        """Compute force-based reward component."""
        reward = torch.zeros_like(forces[:, 0])
        if self.force_scale < 0:  # Negative scale means penalize unwanted contact
            for i in range(forces.shape[1]):
                reward += (1 - desired_contacts[:, i]) * (1 - torch.exp(-forces[:, i] ** 2 / self.force_sigma))
        else:  # Positive scale means reward desired contact
            for i in range(forces.shape[1]):
                reward += (1 - desired_contacts[:, i]) * torch.exp(-forces[:, i] ** 2 / self.force_sigma)

        return (reward / forces.shape[1]) * self.force_scale

    def _compute_velocity_reward(self, velocities: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
        """Compute velocity-based reward component."""
        reward = torch.zeros_like(velocities[:, 0])
        if self.vel_scale < 0:  # Negative scale means penalize movement during contact
            for i in range(velocities.shape[1]):
                reward += desired_contacts[:, i] * (1 - torch.exp(-velocities[:, i] ** 2 / self.vel_sigma))
        else:  # Positive scale means reward movement during swing
            for i in range(velocities.shape[1]):
                reward += desired_contacts[:, i] * torch.exp(-velocities[:, i] ** 2 / self.vel_sigma)

        return (reward / velocities.shape[1]) * self.vel_scale

class ActionSmoothnessPenalty(ManagerTermBase):
    """
    A reward term for penalizing large instantaneous changes in the network action output.
    This penalty encourages smoother actions over time.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward term.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.dt = env.step_dt
        self.prev_prev_action = None
        self.prev_action = None
        # self.__name__ = "action_smoothness_penalty"

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Compute the action smoothness penalty.

        Args:
            env: The RL environment instance.

        Returns:
            The penalty value based on the action smoothness.
        """
        # Get the current action from the environment's action manager
        current_action = env.action_manager.action.clone()

        # If this is the first call, initialize the previous actions
        if self.prev_action is None:
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        if self.prev_prev_action is None:
            self.prev_prev_action = self.prev_action
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        # Compute the smoothness penalty
        penalty = torch.sum(torch.square(current_action - 2 * self.prev_action + self.prev_prev_action), dim=1)

        # Update the previous actions for the next call
        self.prev_prev_action = self.prev_action
        self.prev_action = current_action

        # Apply a condition to ignore penalty during the first few episodes
        startup_env_mask = env.episode_length_buf < 3
        penalty[startup_env_mask] = 0

        # Return the penalty scaled by the configured weight
        return penalty