# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING

from isaac_lab_actuator_dynamic.assets import LEGACTUATORDYNAMIC_CFG, LEGACTUATORDYNAMIC_2_CFG

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")

@configclass
class RandomScaleCfg:
    joint_pos = {
        'L_hip_joint': [-0.1, 0.1],
        'L_hip2_joint': [-0.1, 0.1],
        'L_thigh_joint': [-0.1, 0.1],
        'L_calf_joint': [-0.1, 0.0],
        'L_toe_joint': [-0.1, 0.1],
    }
    joint_vel = {
        'L_hip_joint': [-0.05, 0.05],
        'L_hip2_joint': [-0.05, 0.05],
        'L_thigh_joint': [-0.05, 0.05],
        'L_calf_joint': [-0.05, 0.05],
        'L_toe_joint': [-0.05, 0.05],
    }

@configclass
class ActuatorDynamic2EnvCfg(DirectRLEnvCfg):
    """Actuator Dynamic AMP environment config (base class)."""

    # rewards
    joint_torque_reward_scale = -0.5e-5
    joint_vel_reward_scale = -0.5e-3
    joint_accel_reward_scale = -0.5e-6
    action_rate_reward_scale = -0.5e-4
    action_acc_reward_scale = -0.5e-6
    terminated_scale = -100.0

    alive_scale = 1.0
    joint_pos_mimic_reward_scale = 5.0
    joint_vel_mimic_reward_scale = 3.0
    joint_pos_bonus_reward_scale = 100.0

    # env
    episode_length_s = 10.0
    decimation = 1

    # spaces
    observation_space = 10
    action_space = 5
    state_space = 0

    early_termination = True
    termination_height = 0.5

    # reference motion
    key_dof_names = ["L_hip_joint", "L_hip2_joint", "L_thigh_joint", "L_calf_joint", "L_toe_joint"]
    key_body_names = ["L_hip2", "L_thigh", "L_calf", "L_toe"]

    motion_file: str = [
        os.path.join(MOTIONS_DIR, "recorded_real_motor_data_1.npz"),
        os.path.join(MOTIONS_DIR, "recorded_real_motor_data_2.npz"),
    ]
    # motion_file: str = os.path.join(MOTIONS_DIR, "ik_trajectory_data.npz")
    reference_body = "base"
    reset_strategy = "random"  # default, random, random-start
    """Strategy to be followed when resetting each environment (humanoid's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """

    # reset noise
    randomize_initial_state = True
    random_scale_cfg = RandomScaleCfg()

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.002,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=10.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = LEGACTUATORDYNAMIC_2_CFG.replace(prim_path="/World/envs/env_.*/Robot") # type: ignore

@configclass
class ActuatorDynamic2PlayEnvCfg(ActuatorDynamic2EnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        self.reset_strategy = "random-start"
        self.randomize_initial_state = False
        self.episode_length_s = 30
        self.motion_file = [os.path.join(MOTIONS_DIR, "recorded_real_motor_data_2.npz")]

