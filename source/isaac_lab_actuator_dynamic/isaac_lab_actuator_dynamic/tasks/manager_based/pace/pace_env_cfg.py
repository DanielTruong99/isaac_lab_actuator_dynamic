from curses.ascii import ctrl
import math
from dataclasses import MISSING
import os

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import torch
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ContactSensorCfg,
    LocomotionVelocityRoughEnvCfg,
    ObsTerm,
    RewardsCfg,
    EventCfg,
    ObservationsCfg,
)       
from isaaclab.managers import TerminationTermCfg as DoneTerm

from isaaclab.sim import SimulationCfg
from isaaclab.sim._impl.newton_manager_cfg import NewtonCfg
from isaaclab.sim._impl.solvers_cfg import MJWarpSolverCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import isaaclab.sim as sim_utils

from isaaclab.utils import configclass
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    EventTermCfg,
)
from isaaclab.utils.noise import AdditiveUniformNoiseCfg, AdditiveGaussianNoiseCfg
from isaaclab.envs.mdp.actions import joint_actions
##
# User defined configs
##
from isaac_lab_actuator_dynamic.assets import LEGWALKING_HIGH_GAIN_AMARTURE_2_CFG
from . import mdp as custom_mdp


import torch

import torch
from typing import Optional, Tuple
import torch.nn as nn

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = custom_mdp.CustomJointPositionActionCfg(
        asset_name="robot", 
        joint_names=[
            "L_hip_joint",
            "L_hip2_joint",
            "L_thigh_joint",
            "L_calf_joint",
            "L_toe_joint",
        ], 
        use_default_offset=False,
    )

@configclass
class WalkingRobotObservationsCfg(ObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""
        # joint state
        joint_pos = ObservationTermCfg(func=mdp.joint_pos, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=1.0)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=0.05)
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg() 

    

@configclass 
class WalkingRobotEventCfg:
    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.5, 0.5),
        },
    )


@configclass
class WalkingRobotRewardCfg:
    # dof_torques_l2 = RewardTermCfg(func=mdp.joint_torques_l2, weight=-11.0e-5)
    motion_position_error = RewardTermCfg(
        func=custom_mdp.motion_position_error,
        weight=1.0,
        params={
            "command_name": "motion",
        },
    )
    # motion_velocity_error = RewardTermCfg(
    #     func=custom_mdp.motion_velocity_error,
    #     weight=5.0,
    #     params={
    #         "command_name": "motion",
    #     },
    # )


@configclass
class WalkingRobotCommandsCfg:
    motion = custom_mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        motion_file=[
            # os.path.join(MOTIONS_DIR, "recorded_real_motor_data_1.npz"),
            # os.path.join(MOTIONS_DIR, "recorded_real_motor_data_2.npz"),
            os.path.join(MOTIONS_DIR, "recorded_left_leg_chirp.npz"),
        ]
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class PaceNewtonEnvCfg(LocomotionVelocityRoughEnvCfg):
    observations: WalkingRobotObservationsCfg = WalkingRobotObservationsCfg()
    rewards: WalkingRobotRewardCfg = WalkingRobotRewardCfg()
    events: WalkingRobotEventCfg = WalkingRobotEventCfg()
    commands: WalkingRobotCommandsCfg = WalkingRobotCommandsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    actions: ActionsCfg = ActionsCfg()

    sim: SimulationCfg = SimulationCfg(
        newton_cfg=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=210,
                ncon_per_env=35,
                ls_iterations=10,
                ls_parallel=True,
                cone="pyramidal",
                impratio=1,
                integrator="implicit",
            ),
            num_substeps=1,
            debug_mode=False,
        )
    )
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator.curriculum = False #type: ignore
        self.curriculum.terrain_levels = None

        self.episode_length_s = 3000.0
        self.sim.dt = 0.002
        self.decimation = 1


        self.scene.robot = LEGWALKING_HIGH_GAIN_AMARTURE_2_CFG.replace(prim_path="/World/envs/env_.*/Robot") #type: ignore 

@configclass
class WalkingRobotEvalObservationsCfg(ObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""
        # joint state
        joint_pos = ObservationTermCfg(func=mdp.joint_pos, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=1.0)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=0.05)
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class DebugCfg(ObservationGroupCfg):
        """Observations for debug group."""
        # joint state
        joint_pos = ObservationTermCfg(func=mdp.joint_pos, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=1.0)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=0.05)
        joint_pos_sim_real_hip = ObservationTermCfg(
            func=custom_mdp.joint_pos_sim_real, 
            noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), 
            clip=(-100.0, 100.0), 
            scale=1.0,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", joint_names=["L_hip_joint"]),
            }
        )

        joint_pos_sim_real_hip2 = ObservationTermCfg(
            func=custom_mdp.joint_pos_sim_real, 
            noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), 
            clip=(-100.0, 100.0), 
            scale=1.0,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", joint_names=["L_hip2_joint"]),
            }
        )

        joint_pos_sim_real_thigh = ObservationTermCfg(
            func=custom_mdp.joint_pos_sim_real, 
            noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), 
            clip=(-100.0, 100.0), 
            scale=1.0,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", joint_names=["L_thigh_joint"]),
            }
        )

        joint_pos_sim_real_calf = ObservationTermCfg(
            func=custom_mdp.joint_pos_sim_real, 
            noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), 
            clip=(-100.0, 100.0), 
            scale=1.0,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", joint_names=["L_calf_joint"]),
            }
        )

        joint_pos_sim_real_toe = ObservationTermCfg(
            func=custom_mdp.joint_pos_sim_real, 
            noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), 
            clip=(-100.0, 100.0), 
            scale=1.0,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", joint_names=["L_toe_joint"]),
            }
        )

        joint_effort_sim_real_hip = ObservationTermCfg(
            func=custom_mdp.joint_effort_sim_real, 
            noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), 
            clip=(-100.0, 100.0), 
            scale=1.0,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", joint_names=["L_hip_joint"]),
            }
        )

        joint_effort_sim_real_hip2 = ObservationTermCfg(
            func=custom_mdp.joint_effort_sim_real, 
            noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), 
            clip=(-100.0, 100.0), 
            scale=1.0,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", joint_names=["L_hip2_joint"]),
            }
        )

        joint_effort_sim_real_thigh = ObservationTermCfg(
            func=custom_mdp.joint_effort_sim_real, 
            noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), 
            clip=(-100.0, 100.0), 
            scale=1.0,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", joint_names=["L_thigh_joint"]),
            }
        )

        joint_effort_sim_real_calf = ObservationTermCfg(
            func=custom_mdp.joint_effort_sim_real, 
            noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), 
            clip=(-100.0, 100.0), 
            scale=1.0,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", joint_names=["L_calf_joint"]),
            }
        )

        joint_effort_sim_real_toe = ObservationTermCfg(
            func=custom_mdp.joint_effort_sim_real, 
            noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), 
            clip=(-100.0, 100.0), 
            scale=1.0,
            params={
                "asset_cfg": SceneEntityCfg(name="robot", joint_names=["L_toe_joint"]),
            }
        )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg() 
    debug: DebugCfg = DebugCfg()

@configclass
class WalkingRobotCommandsEvalCfg:
    motion = custom_mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        motion_file=[
            # os.path.join(MOTIONS_DIR, "recorded_real_motor_data_1.npz"),
            # os.path.join(MOTIONS_DIR, "recorded_real_motor_data_2.npz"),
            os.path.join(MOTIONS_DIR, "recorded_left_leg_chirp.npz"),
        ]
    )

@configclass
class PaceNewtonEvalEnvCfg(PaceNewtonEnvCfg):
    observations: WalkingRobotEvalObservationsCfg = WalkingRobotEvalObservationsCfg()
    commands: WalkingRobotCommandsEvalCfg = WalkingRobotCommandsEvalCfg()

    def __post_init__(self):
        super().__post_init__()

        self.events.reset_robot_joints = None
