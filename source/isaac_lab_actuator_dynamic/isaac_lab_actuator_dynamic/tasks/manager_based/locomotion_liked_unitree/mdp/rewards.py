import torch
import isaaclab.envs.mdp as mdp
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg

from ..walking_robot import WalkingRobotEnv

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

    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    base_pos_w = asset.data.root_pos_w
    feet_height = feet_pos_w - base_pos_w.unsqueeze(1) 

    result = ~is_contact * torch.square(feet_height[:, :, 2] - (-0.7405))
    return torch.sum(result, dim=1)
