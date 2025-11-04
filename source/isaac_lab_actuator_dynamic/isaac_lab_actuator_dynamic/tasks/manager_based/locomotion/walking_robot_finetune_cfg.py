from curses.ascii import ctrl
import math
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import torch
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
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

from isaaclab.utils import configclass
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    EventTermCfg,
)
from isaaclab.utils.noise import AdditiveUniformNoiseCfg, AdditiveGaussianNoiseCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs.mdp.actions import joint_actions
##
# User defined configs
##
from isaac_lab_actuator_dynamic.assets import LEGWALKING_HIGH_GAIN_CFG, LEGWALKING_HIGH_GAIN_AMARTURE_CFG
from . import mdp as custom_mdp
from .walking_robot_cfg import WalkingRobotObservationsCfg, WalkingRobotNewtonEnvCfg

import torch

import torch
from typing import Optional, Tuple
import torch.nn as nn

@configclass
class StudentPolicyCfg(ObservationGroupCfg):
    """Observations for policy group."""

    # observation terms (order preserved)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.05), scale=0.25)
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.025),
    )
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01))
    joint_vel = ObsTerm(func=mdp.joint_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), scale=0.05)
    actions = ObsTerm(func=mdp.last_action, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01))

    # vel cmds
    velocity_commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    # gait_phase = ObservationTermCfg(func=custom_mdp.get_gait_phase)
    gait_command = ObservationTermCfg(func=custom_mdp.get_gait_command, params={"command_name": "gait_command"})

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True

# class PolicyCfg(ObservationGroupCfg):
#     """Observations for policy group."""
#     # base state
#     base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.05), clip=(-100.0, 100.0), scale=0.25)
#     base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.05), clip=(-100.0, 100.0), scale=0.25)
#     projected_gravity = ObservationTermCfg(func=mdp.projected_gravity, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.025), clip=(-100.0, 100.0), scale=1.0)
    
#     # joint state
#     joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=1.0)
#     joint_vel = ObservationTermCfg(func=mdp.joint_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=0.05)
    
#     # last action
#     last_action = ObservationTermCfg(func=mdp.last_action, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=1.0)

#     # vel cmds
#     velocity_commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})

#     # gaits
#     gait_phase = ObservationTermCfg(func=custom_mdp.get_gait_phase)
#     gait_command = ObservationTermCfg(func=custom_mdp.get_gait_command, params={"command_name": "gait_command"})

#     def __post_init__(self):
#         self.enable_corruption = True
#         self.concatenate_terms = True

@configclass
class StudentObservationsCfg:
    """Observation specifications for the MDP."""

    # observation groups
    policy: StudentPolicyCfg = StudentPolicyCfg()

@configclass
class TeacherStudentObservationsCfg:
    """Observation specifications for the MDP."""

    # the policy is the student policy
    policy: StudentPolicyCfg = StudentPolicyCfg()
    # the teacher gets is the privileged observations
    teacher = WalkingRobotObservationsCfg.PolicyCfg()


@configclass
class WalkingRobot_FlatTeacherStudentEnvCfg(WalkingRobotNewtonEnvCfg):
    observations: TeacherStudentObservationsCfg = TeacherStudentObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

@configclass
class WalkingRobot_FlatStudentEnvCfg(WalkingRobotNewtonEnvCfg):
    observations: StudentObservationsCfg = StudentObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

@configclass
class WalkingRobot_FlatStudentPlayEnvCfg(WalkingRobotNewtonEnvCfg):
    observations: StudentObservationsCfg = StudentObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        
        # self.observations.policy.enable_corruption = False

        # self.events.add_base_mass = 
        # self.events.base_com.params = {
        #     "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        #     "com_range": {"x": (-0.2, -0.2), "y": (-0.00, 0.00), "z": (-0.00, 0.00)},
        # }
        # self.events.add_base_mass = None #type: ignore
        # self.events.robot_physics_material = None #type: ignore
        # self.events.robot_joint_stiffness_and_damping = None #type: ignore
        # # self.events.robot_center_of_mass = None #type: ignoree
        # self.events.reset_robot_joints = None #type: ignore
        # self.events.reset_robot_base = None #type: ignore
        # self.events.randomize_actuator_gains = None #type: ignore
        # self.events.push_robot = None #type: ignore
        # self.events.reset_base = None #type: ignore
        # # self.events.reset_robot_joints = None #type: ignore
        # self.events.reset_base.params = {
        #     "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
        #     "velocity_range": {
        #         "x": (0.0, 0.0),
        #         "y": (0.0, 0.0),
        #         "z": (0.0, 0.0),
        #         "roll": (0.0, 0.0),
        #         "pitch": (0.0, 0.0),
        #         "yaw": (0.0, 0.0),
        #     },
        # }

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator.curriculum = False #type: ignore
        self.curriculum.terrain_levels = None #type: ignore

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.00, -0.00)
        # self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.heading_control_stiffness = 3.5
   
        # self.commands.gait_command.ranges.frequencies = (0.01, 0.01)
        # self.commands.gait_command.ranges.offsets = (0.01, 0.01)
        # self.commands.gait_command.ranges.durations = (0.99, 0.99)
        # self.commands.gait_command.ranges.swing_height = (0.0, 0.0)

        self.commands.gait_command.ranges.frequencies = (0.8, 0.8)
        self.commands.gait_command.ranges.offsets = (0.5, 0.5)
        self.commands.gait_command.ranges.durations = (0.5, 0.5)
        self.commands.gait_command.ranges.swing_height = (0.1, 0.1)

        # self.viewer.asset_name = "robot"
        # self.viewer.origin_type = "asset_root"
        # self.viewer.eye = (0.0, 5.0, 2.0)


        # self.sim.use_fabric = False
        # self.sim.device = "cpu"
        # self.actions.joint_pos.scale = {"R_toe_joint": 0.001}
        # self.actions.joint_pos = Actions2PlayCfg()

     
    


    