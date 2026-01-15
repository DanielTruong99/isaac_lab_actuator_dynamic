from curses.ascii import ctrl
import math
from dataclasses import MISSING
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import torch
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuatorCfg
# from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
#     LocomotionVelocityRoughEnvCfg,
#     ObsTerm,
#     RewardsCfg,
#     EventCfg,
#     ObservationsCfg,
# )       
from isaaclab.envs import ManagerBasedRLEnvCfg
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
import isaaclab.sim as sim_utils
from isaaclab.utils.noise import AdditiveUniformNoiseCfg, AdditiveGaussianNoiseCfg
from isaaclab.envs.mdp.actions import joint_actions
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
##
# User defined configs
##
from isaac_lab_actuator_dynamic.assets import LEGWALKING_HIGH_GAIN_AMARTURE_3_CFG, LEGWALKING_HIGH_GAIN_AMARTURE_CFG, LEGWALKING_HIGH_GAIN_AMARTURE_5_CFG, LEGWALKING_HIGH_GAIN_AMARTURE_5_NEWFOOT_CFG
from . import mdp as custom_mdp
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
import torch
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from typing import Optional, Tuple
import torch.nn as nn

from .walking_robot_cfg import WalkingRobotNewtonEnvCfg


@configclass
class TeacherStudentObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""
        # base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.05), clip=(-100.0, 100.0), scale=0.25)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.12), clip=(-100.0, 100.0), scale=0.25)
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.025), clip=(-100.0, 100.0), scale=1.0)
        
        # joint state
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=1.0)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.35), clip=(-100.0, 100.0), scale=0.05)
        
        # last action
        last_action = ObservationTermCfg(func=mdp.last_action, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=1.0)

        # vel cmds
        velocity_commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # gaits
        gait_phase = ObservationTermCfg(func=custom_mdp.get_gait_phase)
        gait_command = ObservationTermCfg(func=custom_mdp.get_gait_command, params={"command_name": "gait_command"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = True

    @configclass
    class PlayObsCfg(ObservationGroupCfg):
        """Observations for policy group."""
        # base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.05), clip=(-100.0, 100.0), scale=0.25)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.12), clip=(-100.0, 100.0), scale=0.25)
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.025), clip=(-100.0, 100.0), scale=1.0)
        
        # joint state
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=1.0)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.35), clip=(-100.0, 100.0), scale=0.05)
        
        # last action
        last_action = ObservationTermCfg(func=mdp.last_action, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=1.0)

        # vel cmds
        velocity_commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # gaits
        gait_phase = ObservationTermCfg(func=custom_mdp.get_gait_phase)
        gait_command = ObservationTermCfg(func=custom_mdp.get_gait_command, params={"command_name": "gait_command"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            # self.history_length = 10
            # self.flatten_history_dim = True

    @configclass
    class TeacherPolicyCfg(ObservationGroupCfg):
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.05), clip=(-100.0, 100.0), scale=0.25)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.12), clip=(-100.0, 100.0), scale=0.25)
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.025), clip=(-100.0, 100.0), scale=1.0)
        
        # joint state
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=1.0)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.35), clip=(-100.0, 100.0), scale=0.05)
        
        # last action
        last_action = ObservationTermCfg(func=mdp.last_action, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=1.0)

        # vel cmds
        velocity_commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # gaits
        gait_phase = ObservationTermCfg(func=custom_mdp.get_gait_phase)
        gait_command = ObservationTermCfg(func=custom_mdp.get_gait_command, params={"command_name": "gait_command"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            # self.history_length = 10
            # self.flatten_history_dim = True



    @configclass
    class CriticCfg(ObservationGroupCfg):
        """Observations for policy group."""
        # base state
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel)
        proj_gravity = ObservationTermCfg(func=mdp.projected_gravity)

        # joint state
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel)

        # last action
        last_action = ObservationTermCfg(func=mdp.last_action)

        # vel cmds
        velocity_commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # gaits
        gait_phase = ObservationTermCfg(func=custom_mdp.get_gait_phase)
        gait_command = ObservationTermCfg(func=custom_mdp.get_gait_command, params={"command_name": "gait_command"})

        # simulation states
        robot_base_pos = ObservationTermCfg(func=mdp.root_pos_w)
        robot_base_quat = ObservationTermCfg(func=mdp.root_quat_w)
        robot_base_lin_vel = ObservationTermCfg(func=mdp.root_lin_vel_w)
        robot_base_ang_vel = ObservationTermCfg(func=mdp.root_ang_vel_w)

        robot_joint_torque = ObservationTermCfg(func=custom_mdp.joint_torque)
        robot_joint_acc = ObservationTermCfg(func=custom_mdp.joint_acc)
        feet_lin_vel = ObservationTermCfg(
            func=custom_mdp.feet_lin_vel, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_toe")}
        )
        robot_mass = ObservationTermCfg(func=custom_mdp.robot_mass)
        robot_inertia = ObservationTermCfg(func=custom_mdp.robot_inertia)
        robot_joint_pos = ObservationTermCfg(func=custom_mdp.robot_joint_pos)
        robot_joint_stiffness = ObservationTermCfg(func=custom_mdp.robot_joint_stiffness)
        robot_joint_damping = ObservationTermCfg(func=custom_mdp.robot_joint_damping)
        robot_pos = ObservationTermCfg(func=custom_mdp.robot_pos)
        robot_vel = ObservationTermCfg(func=custom_mdp.robot_vel)
        # robot_material_properties = ObservationTermCfg(func=custom_mdp.robot_material_properties)
        feet_contact_force = ObservationTermCfg(
            func=custom_mdp.robot_contact_force, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe")}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True   


    # observation groups
    teacher: TeacherPolicyCfg = TeacherPolicyCfg() 
    # critic: CriticCfg = CriticCfg()
    policy: PolicyCfg = PolicyCfg()
    play_obs: PlayObsCfg = PlayObsCfg()

@configclass
class WalkingRobot_TeacherStudentEnvCfg(WalkingRobotNewtonEnvCfg):
    observations: TeacherStudentObservationsCfg = TeacherStudentObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

@configclass
class WalkingRobot_TeacherStudentPlayEnvCfg(WalkingRobot_TeacherStudentEnvCfg):
    

    def __post_init__(self) -> None:
        super().__post_init__()

        self.events.push_robot = None
        self.events.reset_robot_base = None
        self.events.reset_robot_joints = None
        self.events.robot_joint_stiffness_and_damping = None
        self.events.reset_robot_states = None

@configclass
class StudentObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""
        # base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.05), clip=(-100.0, 100.0), scale=0.25)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.12), clip=(-100.0, 100.0), scale=0.25)
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.025), clip=(-100.0, 100.0), scale=1.0)
        
        # joint state
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=1.0)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.35), clip=(-100.0, 100.0), scale=0.05)
        
        # last action
        last_action = ObservationTermCfg(func=mdp.last_action, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=1.0)

        # vel cmds
        velocity_commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # gaits
        gait_phase = ObservationTermCfg(func=custom_mdp.get_gait_phase)
        gait_command = ObservationTermCfg(func=custom_mdp.get_gait_command, params={"command_name": "gait_command"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = True

    @configclass
    class PlayObsCfg(ObservationGroupCfg):
        """Observations for policy group."""
        # base state
        # base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.05), clip=(-100.0, 100.0), scale=0.25)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.12), clip=(-100.0, 100.0), scale=0.25)
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.025), clip=(-100.0, 100.0), scale=1.0)
        
        # joint state
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=1.0)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.2), clip=(-100.0, 100.0), scale=0.05)
        
        # last action
        last_action = ObservationTermCfg(func=mdp.last_action, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=1.0)

        # vel cmds
        velocity_commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # gaits
        # gait_phase = ObservationTermCfg(func=custom_mdp.get_gait_phase)
        gait_command = ObservationTermCfg(func=custom_mdp.get_gait_command, params={"command_name": "gait_command"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class TeacherPolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""
        # base state
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.05), clip=(-100.0, 100.0), scale=0.25)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.12), clip=(-100.0, 100.0), scale=0.25)
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.025), clip=(-100.0, 100.0), scale=1.0)
        
        # joint state
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=1.0)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.2), clip=(-100.0, 100.0), scale=0.05)
        
        # last action
        last_action = ObservationTermCfg(func=mdp.last_action, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=1.0)

        # vel cmds
        velocity_commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # gaits
        gait_phase = ObservationTermCfg(func=custom_mdp.get_gait_phase)
        gait_command = ObservationTermCfg(func=custom_mdp.get_gait_command, params={"command_name": "gait_command"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = True



    @configclass
    class CriticCfg(ObservationGroupCfg):
        """Observations for policy group."""
        # base state
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel)
        proj_gravity = ObservationTermCfg(func=mdp.projected_gravity)

        # joint state
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel)

        # last action
        last_action = ObservationTermCfg(func=mdp.last_action)

        # vel cmds
        velocity_commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # gaits
        gait_phase = ObservationTermCfg(func=custom_mdp.get_gait_phase)
        gait_command = ObservationTermCfg(func=custom_mdp.get_gait_command, params={"command_name": "gait_command"})

        # simulation states
        robot_base_pos = ObservationTermCfg(func=mdp.root_pos_w)
        robot_base_quat = ObservationTermCfg(func=mdp.root_quat_w)
        robot_base_lin_vel = ObservationTermCfg(func=mdp.root_lin_vel_w)
        robot_base_ang_vel = ObservationTermCfg(func=mdp.root_ang_vel_w)

        robot_joint_torque = ObservationTermCfg(func=custom_mdp.joint_torque)
        robot_joint_acc = ObservationTermCfg(func=custom_mdp.joint_acc)
        feet_lin_vel = ObservationTermCfg(
            func=custom_mdp.feet_lin_vel, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_toe")}
        )
        robot_mass = ObservationTermCfg(func=custom_mdp.robot_mass)
        robot_inertia = ObservationTermCfg(func=custom_mdp.robot_inertia)
        robot_joint_pos = ObservationTermCfg(func=custom_mdp.robot_joint_pos)
        robot_joint_stiffness = ObservationTermCfg(func=custom_mdp.robot_joint_stiffness)
        robot_joint_damping = ObservationTermCfg(func=custom_mdp.robot_joint_damping)
        robot_pos = ObservationTermCfg(func=custom_mdp.robot_pos)
        robot_vel = ObservationTermCfg(func=custom_mdp.robot_vel)
        # robot_material_properties = ObservationTermCfg(func=custom_mdp.robot_material_properties)
        feet_contact_force = ObservationTermCfg(
            func=custom_mdp.robot_contact_force, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe")}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True   


    # observation groups
    # teacher: TeacherPolicyCfg = TeacherPolicyCfg() 
    critic: CriticCfg = CriticCfg()
    policy: PolicyCfg = PolicyCfg()
    # play_obs: PlayObsCfg = PlayObsCfg()

@configclass
class WalkingRobot_StudentEnvCfg(WalkingRobotNewtonEnvCfg):
    observations: StudentObservationsCfg = StudentObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

