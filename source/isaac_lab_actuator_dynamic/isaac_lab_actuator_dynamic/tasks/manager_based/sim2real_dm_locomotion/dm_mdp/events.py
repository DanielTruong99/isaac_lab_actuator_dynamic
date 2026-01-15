from __future__ import annotations

# from pyglet import env

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

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
    from isaaclab.envs import ManagerBasedEnv

def reset_robot_state_by_motion(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range,
    velocity_range,
    joint_position_range,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints with offsets around the default position and velocity by the given ranges.

    This function samples random values from the given ranges and biases the default joint positions and velocities
    by these values. The biased values are then set into the physics simulation.
    """
    motion_command = env.command_manager._terms["motion"]
    asset = env.scene[asset_cfg.name]

    motion_command.motion.resample_motions(env_ids)
    motion_command._uniform_sampling(env_ids)

    root_pos = motion_command.body_pos_w[:, 0].clone()
    root_ori = motion_command.body_quat_w[:, 0].clone()
    root_lin_vel = motion_command.body_lin_vel_w[:, 0].clone()
    root_ang_vel = motion_command.body_ang_vel_w[:, 0].clone()
    # root_lin_vel = torch.zeros_like(root_lin_vel)
    root_ang_vel = torch.zeros_like(root_ang_vel)

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)
    root_pos[env_ids] += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)
    root_lin_vel[env_ids] += rand_samples[:, :3]
    root_ang_vel[env_ids] += rand_samples[:, 3:]

    joint_pos = motion_command.joint_pos.clone()
    joint_vel = motion_command.joint_vel.clone()
    # joint_vel = torch.zeros_like(joint_vel)

    joint_pos += sample_uniform(*joint_position_range, joint_pos.shape, joint_pos.device)
    soft_joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos[env_ids] = torch.clip(
        joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
    )
    asset.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
    asset.write_root_state_to_sim(
        torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
        env_ids=env_ids,
    )

def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)