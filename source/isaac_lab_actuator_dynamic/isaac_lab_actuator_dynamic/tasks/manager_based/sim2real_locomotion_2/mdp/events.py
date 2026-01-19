
from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
##
#! User defined configs
##
from ..walking_robot import WalkingRobotEnv
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

import warp as wp

def reset_robot_states(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    root_pose_range: dict[str, tuple[float, float]],
    root_velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot to its default joint positions and velocities.

    This function retrieves the default joint positions and velocities from the asset configuration
    and sets them into the physics simulation for the specified environment IDs.
    """
    # 50% chance to randomize root pose around default pose
    is_zeros_state = torch.multinomial(torch.tensor([0.5, 0.5]), len(env_ids), replacement=True)
    env_ids_default = env_ids[is_zeros_state == 0]
    env_ids_zeros = env_ids[is_zeros_state == 1]
 
    if len(env_ids_default) != 0:
        # reset root pose with offsets
        mdp.reset_root_state_uniform(
            env,
            env_ids_default,
            root_pose_range,
            root_velocity_range,
            asset_cfg=asset_cfg,
        )

        # reset joint states with offsets
        mdp.reset_joints_by_offset(
            env,
            env_ids_default,
            position_range,
            velocity_range,
            asset_cfg=asset_cfg,
        )
    
    if len(env_ids_zeros) != 0:
 
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]

        # get default joint state are zeroed out
        joint_pos = asset.data.default_joint_pos[env_ids_zeros, asset_cfg.joint_ids].clone()
        joint_pos = torch.zeros_like(joint_pos)
        joint_vel = asset.data.default_joint_vel[env_ids_zeros, asset_cfg.joint_ids].clone()

        # bias these values randomly
        joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
        joint_vel += math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

        # clamp joint pos to limits
        joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids_zeros, asset_cfg.joint_ids]
        
        joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
        # clamp joint vel to limits
        joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids_zeros, asset_cfg.joint_ids]
        joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

        # set into the physics simulation
        asset.write_joint_state_to_sim(
            joint_pos.view(len(env_ids_zeros), -1),
            joint_vel.view(len(env_ids_zeros), -1),
            env_ids=env_ids_zeros,
            joint_ids=asset_cfg.joint_ids,
        )

        
        # get default root state
        root_states = asset.data.default_root_state[env_ids_zeros].clone()
        root_states[:, 2] = 0.65

        # poses
        range_list = [root_pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids_zeros), 6), device=asset.device)

        positions = root_states[:, 0:3] + env.scene.env_origins[env_ids_zeros] + rand_samples[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
        # velocities
        range_list = [root_velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids_zeros), 6), device=asset.device)

        velocities = root_states[:, 7:13] + rand_samples

        # set into the physics simulation
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids_zeros)
        asset.write_root_velocity_to_sim(velocities, env_ids=env_ids_zeros)

def randomize_rigid_body_coms(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_distribution_params: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the center of mass (COM) of the bodies by adding, scaling, or setting random values for each dimension.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    coms = asset.root_physx_view.get_coms().clone()

    # Apply randomization to each dimension separately
    for dim in range(3):  # 0=x, 1=y, 2=z
        coms[..., dim] = _randomize_prop_by_op(
            coms[..., dim],
            com_distribution_params[dim],
            env_ids,
            body_ids,
            operation=operation,
            distribution=distribution,
        )

    asset.root_physx_view.set_coms(coms, env_ids)

def update_phase(env: WalkingRobotEnv, env_ids: torch.Tensor):
    """
    Update the phase of the walking robot environment.
    This function calculates and updates the phase of the walking robot's gait cycle.
    The phase is determined based on the episode length buffer and the step duration.
    It also calculates the phase for the left and right legs of the robot.
    Args:
        env (WalkingRobotEnv): The walking robot environment instance.
    Attributes:
        env.phase (float): The current phase of the robot's gait cycle.
        env.phase_left (float): The phase of the left leg.
        env.phase_right (float): The phase of the right leg.
    """

    period = 0.8
    offset = 0.5
    env.phase[env_ids] = (env.episode_length_buf[env_ids] * env.step_dt) % period / period
    env.phase_left[env_ids] = env.phase[env_ids]
    env.phase_right[env_ids] = (env.phase[env_ids] + offset) % 1

def randomize_rigid_body_mass_inertia(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass_inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the inertia of the bodies by adding, scaling, or setting random values.

    This function allows randomizing the mass of the bodies of the asset. The function samples random values from the
    given distribution parameters and adds, scales, or sets the values into the physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body masses. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]

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

    # get the current inertias of the bodies (num_assets, num_bodies)
    inertias = asset.root_physx_view.get_inertias().clone()
    masses = asset.root_physx_view.get_masses().clone()

    masses = _randomize_prop_by_op(
        masses, mass_inertia_distribution_params, env_ids, body_ids, operation=operation, distribution=distribution
    )
    scale = masses / asset.root_physx_view.get_masses()
    inertias *= scale.unsqueeze(-1)

    asset.root_physx_view.set_masses(masses, env_ids)
    asset.root_physx_view.set_inertias(inertias, env_ids)

def apply_external_force_torque_stochastic(
    env,
    env_ids: torch.Tensor,
    force_range: dict[str, tuple[float, float]],
    torque_range: dict[str, tuple[float, float]],
    probability: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize the external forces and torques applied to the bodies.

    This function creates a set of random forces and torques sampled from the given ranges. The number of forces
    and torques is equal to the number of bodies times the number of environments. The forces and torques are
    applied to the bodies by calling ``asset.set_external_force_and_torque``. The forces and torques are only
    applied when ``asset.write_data_to_sim()`` is called in the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # clear the existing forces and torques
    asset._external_force_b *= 0
    asset._external_torque_b *= 0

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    random_values = torch.rand(env_ids.shape, device=env_ids.device)
    mask = random_values < probability
    masked_env_ids = env_ids[mask]

    if len(masked_env_ids) == 0:
        return

    # resolve number of bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    # sample random forces and torques
    size = (len(masked_env_ids), num_bodies, 3)
    force_range_list = [force_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    force_range = torch.tensor(force_range_list, device=asset.device)
    forces = math_utils.sample_uniform(force_range[:, 0], force_range[:, 1], size, asset.device)
    torque_range_list = [torque_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    torque_range = torch.tensor(torque_range_list, device=asset.device)
    torques = math_utils.sample_uniform(torque_range[:, 0], torque_range[:, 1], size, asset.device)
    # set the forces and torques into the buffers
    # note: these are only applied when you call: `asset.write_data_to_sim()`
    asset.set_external_force_and_torque(forces, torques, env_ids=masked_env_ids, body_ids=asset_cfg.body_ids)


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data