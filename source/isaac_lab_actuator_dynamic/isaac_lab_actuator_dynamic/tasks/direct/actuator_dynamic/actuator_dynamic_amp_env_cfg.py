# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING

from isaac_lab_actuator_dynamic.assets import LEGACTUATORDYNAMIC_CFG

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class ActuatorDynamicEnvCfg(DirectRLEnvCfg):
    """Actuator Dynamic AMP environment config (base class)."""

    # rewards
    joint_torque_reward_scale = -2.5e-7
    joint_accel_reward_scale = 0.0
    action_rate_reward_scale = -1e-3
    terminated_scale = -10.0
    alive_scale = 0.0
    joint_torque_mimic_reward_scale = 3.0

    # env
    episode_length_s = 10.0
    decimation = 1

    # spaces
    observation_space = 2
    action_space = 1
    state_space = 0
    num_amp_observations = 2
    amp_observation_space = 2

    early_termination = True
    termination_height = 0.5

    # motion_file: str = [
    #     os.path.join(MOTIONS_DIR, "recorded_motor_data_1.npz"),
    #     os.path.join(MOTIONS_DIR, "recorded_motor_data_2.npz"),
    # ]
    motion_file: str = os.path.join(MOTIONS_DIR, "recorded_motor_data_1.npz")
    reference_body = "base"
    reset_strategy = "random"  # default, random, random-start
    """Strategy to be followed when resetting each environment (humanoid's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=10.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = LEGACTUATORDYNAMIC_CFG.replace(prim_path="/World/envs/env_.*/Robot") # type: ignore

