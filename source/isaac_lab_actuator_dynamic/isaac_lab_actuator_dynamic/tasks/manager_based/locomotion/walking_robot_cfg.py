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
from isaaclab.envs.mdp.actions import joint_actions
##
# User defined configs
##
from isaac_lab_actuator_dynamic.assets import LEGWALKING_HIGH_GAIN_CFG, LEGWALKING_HIGH_GAIN_AMARTURE_CFG
from . import mdp as custom_mdp

import torch

import torch
from typing import Optional, Tuple
import torch.nn as nn



@configclass
class Actions2PlayCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*"], 
        use_default_offset=True,
        scale = {
            ".*_hip_joint": 0.2625,
            ".*_hip2_joint": 0.2625,
            ".*_thigh_joint": 0.2625,
            ".*_calf_joint": 0.1325,
            ".*_toe_joint": 0.35,
        }
    )

@configclass
class WalkingRobotObservationsCfg(ObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""
        # base state
        # base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.05), clip=(-100.0, 100.0), scale=0.25)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.05), clip=(-100.0, 100.0), scale=0.25)
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.025), clip=(-100.0, 100.0), scale=1.0)
        
        # joint state
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=1.0)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=0.05)
        
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
        heights = ObservationTermCfg(func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")})

        robot_joint_torque = ObsTerm(func=custom_mdp.joint_torque)
        robot_joint_acc = ObsTerm(func=custom_mdp.joint_acc)
        feet_lin_vel = ObsTerm(
            func=custom_mdp.feet_lin_vel, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_toe")}
        )
        robot_mass = ObsTerm(func=custom_mdp.robot_mass)
        robot_inertia = ObsTerm(func=custom_mdp.robot_inertia)
        robot_joint_pos = ObsTerm(func=custom_mdp.robot_joint_pos)
        robot_joint_stiffness = ObsTerm(func=custom_mdp.robot_joint_stiffness)
        robot_joint_damping = ObsTerm(func=custom_mdp.robot_joint_damping)
        robot_pos = ObsTerm(func=custom_mdp.robot_pos)
        robot_vel = ObsTerm(func=custom_mdp.robot_vel)
        robot_material_properties = ObsTerm(func=custom_mdp.robot_material_properties)
        feet_contact_force = ObsTerm(
            func=custom_mdp.robot_contact_force, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_toe")}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class DebugCfg(ObservationGroupCfg):
        """Observations for debug group."""
        joint_torque = ObservationTermCfg(func=custom_mdp.joint_torque)
        base_height = ObservationTermCfg(func=mdp.base_pos_z)
        joint_acc = ObservationTermCfg(func=custom_mdp.joint_acc)
        joint_pos = ObservationTermCfg(func=mdp.joint_pos)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel)
        # joint_cmd_pos = ObservationTermCfg(func=custom_mdp.joint_pos_and_cmd)
        # joint_cmd_pos_error = ObservationTermCfg(func=custom_mdp.joint_vel_and_cmd_error)


        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg() 
    critic: CriticCfg = CriticCfg()
    #! Just for debugging
    debug: DebugCfg = DebugCfg()
    

@configclass 
class WalkingRobotEventCfg:

    # startup
    add_base_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )


    base_com = EventTermCfg(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.075, 0.075), "y": (-0.075, 0.075), "z": (-0.075, 0.075)},
        },
    )
    # add_link_mass = EventTermCfg(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=[".*_thigh", ".*_hip", ".*_hip2", ".*_calf", ".*_toe"]),
    #         "mass_distribution_params": (0.8, 1.2),
    #         "operation": "scale",
    #     },
    # )
    # radomize_rigid_body_mass_inertia = EventTermCfg(
    #     func=mdp.randomize_rigid_body_mass_inertia,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "mass_inertia_distribution_params": (0.8, 1.2),
    #         "operation": "scale",
    #     },
    # )
    robot_physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.7, 0.9),
            "restitution_range": (0.0, 1.0),
            "num_buckets": 48,
        },
    )
    # robot_joint_stiffness_and_damping = EventTermCfg(
    #     func=mdp.randomize_actuator_gains,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #         "stiffness_distribution_params": (45.0, 60.0),
    #         "damping_distribution_params": (2.0, 3.0),
    #         "operation": "abs",
    #         "distribution": "uniform",
    #     },
    # )
    # robot_center_of_mass = EventTermCfg(
    #     func=mdp.randomize_rigid_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "com_distribution_params": ((-0.075, 0.075), (-0.075, 0.075), (-0.075, 0.075)),
    #         "operation": "add",
    #         "distribution": "uniform",
    #     },
    # )

    # reset
    reset_robot_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.5, 0.5),
        },
    )

    # reset_robot_hip_joint = EventTermCfg(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["L_hip_joint", "R_hip_joint"]),
    #         "position_range": (-0.1, 0.1),
    #         "velocity_range": (-0.2, 0.2),
    #     },
    # )

    # reset_robot_hip2_joint = EventTermCfg(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["L_hip2_joint", "R_hip2_joint"]),
    #         "position_range": (-0.1, 0.1),
    #         "velocity_range": (-0.1, 0.1),
    #     },
    # )

    # reset_robot_thigh_joint = EventTermCfg(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["L_thigh_joint", "R_thigh_joint"]),
    #         "position_range": (-0.5, 0.2),
    #         "velocity_range": (-0.5, 0.5),
    #     },
    # )

    # reset_robot_calf_joint = EventTermCfg(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["L_calf_joint", "R_calf_joint"]),
    #         "position_range": (-0.1, 0.2),
    #         "velocity_range": (-0.5, 0.5),
    #     },
    # )

    # reset_robot_toe_joint = EventTermCfg(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["L_toe_joint", "R_toe_joint"]),
    #         "position_range": (-0.3, 0.7),
    #         "velocity_range": (-0.5, 0.5),
    #     },
    # )

    # randomize_actuator_gains = EventTermCfg(
    #     func=mdp.randomize_actuator_gains,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #         "stiffness_distribution_params": (0.5, 2.0),
    #         "damping_distribution_params": (0.5, 2.0),
    #         "operation": "scale",
    #         "distribution": "log_uniform",
    #     },
    # )


    push_robot = EventTermCfg(
        func=custom_mdp.apply_external_force_torque_stochastic,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": {
                "x": (-500.0, 500.0),
                "y": (-500.0, 500.0),
                "z": (-0.0, 0.0),
            },  # force = mass * dv / dt
            "torque_range": {"x": (-50.0, 50.0), "y": (-50.0, 50.0), "z": (-0.0, 0.0)},
            "probability": 0.002,  # Expect step = 1 / probability
        },
    )


@configclass
class WalkingRobotRewardCfg:
    track_lin_vel_xy_exp = RewardTermCfg(
        func=mdp.track_lin_vel_xy_exp, weight=2.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewardTermCfg(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    is_alive = RewardTermCfg(
        func=mdp.is_alive,
        weight=1.0,
    )

    gait_reward = RewardTermCfg(
        func=custom_mdp.GaitReward,
        weight=1.5,
        params={
            "tracking_contacts_shaped_force": -2.0,
            "tracking_contacts_shaped_vel": -2.0,
            "gait_force_sigma": 25.0,
            "gait_vel_sigma": 0.25,
            "kappa_gait_probs": 0.05,
            "command_name": "gait_command",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_toe"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_toe"]),
        },
    )

    # hip_2_joint = RewardTermCfg(
    #     func=custom_mdp.hip2_joint_deviation,
    #     weight=-0.7,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip2_joint"])
    #     },
    # )

    # penalty terms
    # termination_penalty = RewardTermCfg(func=mdp.is_terminated, weight=-100.0)
    dof_torques_l2 = RewardTermCfg(func=mdp.joint_torques_l2, weight=-11.0e-5)
    base_height_l2 = RewardTermCfg(
        func=mdp.base_height_l2,
        weight=-50.0,
        params={"target_height": 0.77},
    )
    dof_vel = RewardTermCfg(
        func=mdp.joint_vel_l2,
        weight=-5.0e-05,
    )
    dof_acc_l2 = RewardTermCfg(func=mdp.joint_acc_l2, weight=-1e-6)
    lin_vel_z_l2 = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05)
    action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.5)
    dof_pos_limits = RewardTermCfg(func=mdp.joint_pos_limits, weight=-2.0)
    joint_deviation = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_hip2_joint"])},
    )
    # action_limits = RewardTermCfg(func=custom_mdp.action_limits, weight=-2.0)
    undesired_contacts = RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_hip", ".*_hip2", "base"]), "threshold": 10.0},
    )
    pen_action_smoothness = RewardTermCfg(func=custom_mdp.ActionSmoothnessPenalty, weight=-0.15)
    flat_orientation_l2 = RewardTermCfg(func=mdp.flat_orientation_l2, weight=-5.0)
    pen_feet_distance = RewardTermCfg(
        func=custom_mdp.feet_distance,
        weight=-100,
        params={"min_feet_distance": 0.123,"feet_links_name": ["L_toe", "R_toe"]}
    )
    pen_feet_regulation = RewardTermCfg(
        func=custom_mdp.feet_regulation,
        weight=-0.05, # -0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["L_toe", "R_toe"]),
                "base_height_target": 0.77, "foot_radius": 0.035},
    )

    pen_joint_power_l1 = RewardTermCfg(func=custom_mdp.joint_powers_l1, weight=-2e-5)

    pen_foot_landing_vel = RewardTermCfg(
            func=custom_mdp.foot_landing_vel,
            weight=-0.15,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["L_toe", "R_toe"]),
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["L_toe", "R_toe"]),
                    "foot_radius": 0.035, "about_landing_threshold": 0.08},
        )

    # stand_still_contact = RewardTermCfg(
    #     func=custom_mdp.stand_still_contact,
    #     weight=-0.7,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=["L_toe", "R_toe"]),
    #     },
    # )

    # stand_still = RewardTermCfg(
    #     func=custom_mdp.stand_still,
    #     weight=-0.7,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot")
    #     },
    # )

class CustomUniformVelocityCommand(mdp.UniformVelocityCommand):
    """Custom uniform velocity command generator configuration."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self.metrics["num_standing_envs"] = torch.zeros(self.num_envs, device=self.device)

    def _update_metrics(self):
        super()._update_metrics()
        # -- metrics
        self.metrics["num_standing_envs"] = self.is_standing_env.float()



@configclass
class WalkingRobotCommandsCfg:
    """Command specifications for the MDP."""
    gait_command = custom_mdp.UniformGaitCommandCfg(
        resampling_time_range=(5.0, 5.0),  # Fixed resampling time of 5 seconds
        debug_vis=False,  # No debug visualization needed
        ranges=custom_mdp.UniformGaitCommandCfg.Ranges(
            frequencies=(0.8, 1.6), # (1.5, 2.5),  # Gait frequency range [Hz]
            offsets=(0.5, 0.5),  # Phase offset range [0-1]
            durations=(0.5, 0.5),  # Contact duration range [0-1]
            swing_height=(0.1, 0.2)
        ),
    )

    base_velocity = mdp.UniformVelocityCommandCfg(
        class_type=CustomUniformVelocityCommand,
        asset_name="robot",
        resampling_time_range=(3.0, 15.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=1.0,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.5), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    # base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"asset_cfg": SceneEntityCfg("robot"), "minimum_height": 0.3},)

@configclass
class WalkingRobotEnvCfg(LocomotionVelocityRoughEnvCfg):
    observations: WalkingRobotObservationsCfg = WalkingRobotObservationsCfg()
    rewards: WalkingRobotRewardCfg = WalkingRobotRewardCfg()
    events: WalkingRobotEventCfg = WalkingRobotEventCfg()
    commands: WalkingRobotCommandsCfg = WalkingRobotCommandsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    actions: Actions2PlayCfg = Actions2PlayCfg()

    def __post_init__(self):
        super().__post_init__()

        ''' #!Terrain setup'''
        # self.scene.terrain.terrain_generator = custom_mdp.TERRAINS_CFG
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator.curriculum = False #type: ignore
        self.curriculum.terrain_levels = None
        self.sim.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.decimation = 4

        
        self.scene.robot = LEGWALKING_HIGH_GAIN_AMARTURE_CFG.replace(prim_path="/World/envs/env_.*/Robot") #type: ignore 
        # self.scene.height_scanner = None #type: ignore

        # self.events.add_base_mass = None #type: ignore
        # self.events.robot_physics_material = None #type: ignore
        # self.events.robot_joint_stiffness_and_damping = None #type: ignore
        # self.events.robot_center_of_mass = None #type: ignore
        # self.events.randomize_actuator_gains = None #type: ignore

@configclass
class WalkingRobotNewtonEnvCfg(WalkingRobotEnvCfg):
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

        self.observations.critic.heights = None #type: ignore
        self.observations.critic.robot_material_properties = None #type: ignore
        self.observations.critic.robot_inertia = None #type: ignore
        self.observations.critic.robot_mass = None #type: ignore

        # # self.events.add_base_mass = None #type: ignore
        # # self.events.robot_physics_material = None #type: ignore
        # # self.events.robot_center_of_mass = None #type: ignore
        self.events.robot_joint_stiffness_and_damping = None #type: ignore

        # self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*_hip_joint", ".*_hip2_joint",".*_calf_joint", ".*toe_joint"]
        # )

@configclass
class WalkingRobotNewtonPlayEnvCfg(WalkingRobotNewtonEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # self.commands.base_velocity = None #type: ignore
        # self.commands.gait_command = None #type: ignore

        # # self.sim.render_interval = 8
        # self.episode_length_s = 20.0
        # self.sim.dt = 0.001
        # self.decimation = 20
        # self.sim.render_interval = self.decimation

        # self.observations.policy.enable_corruption = False

        # self.events.add_base_mass = 
        # self.events.base_com.params = {
        #     "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        #     "com_range": {"x": (-0.2, -0.2), "y": (-0.00, 0.00), "z": (-0.00, 0.00)},
        # }
        # self.events.add_base_mass = None #type: ignore
        self.events.robot_physics_material = None #type: ignore
        # self.events.robot_joint_stiffness_and_damping = None #type: ignore
        # self.events.robot_center_of_mass = None #type: ignoree
        self.events.reset_robot_joints = None #type: ignore
        self.events.reset_robot_base = None #type: ignore
        # self.events.randomize_actuator_gains = None #type: ignore
        self.events.push_robot = None #type: ignore
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
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.heading_control_stiffness = 3.5
        self.commands.base_velocity.rel_standing_envs = 0.5

        self.commands.gait_command.ranges.frequencies = (1.5, 1.5)
        self.commands.gait_command.ranges.offsets = (0.5, 0.5)
        self.commands.gait_command.ranges.durations = (0.5, 0.5)
        self.commands.gait_command.ranges.swing_height = (0.1, 0.1)

    


@configclass
class WalkingRobotRewardFinetuneCfg(WalkingRobotRewardCfg):
    pen_knee_power_l1 = RewardTermCfg(
        func=custom_mdp.knee_joint_powers_l1, 
        weight=-2e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_calf_joint"])}
    )

    # joint_deviation = RewardTermCfg(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip2_joint"])},
    # )

    pen_feet_regulation = RewardTermCfg(
        func=custom_mdp.feet_regulation,
        weight=-0.2, # -0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["L_toe", "R_toe"]),
                "base_height_target": 0.83, "foot_radius": 0.05},
    )

    pen_knee_torque_standstill = RewardTermCfg(
        func=custom_mdp.knee_joint_torques_standstill, 
        weight=-3e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_calf_joint"])}
    )

    base_height_l2 = RewardTermCfg(
        func=mdp.base_height_l2,
        weight=-50.0,
        params={"target_height": 0.83},
    )

    def __post_init__(self):
        super().__post_init__()

        # self.base_height_l2 = None #type: ignore
        self.dof_pos_limits = None #type: ignore

@configclass
class WalkingRobotEnvFinetuneCfg(WalkingRobotEnvCfg):
    rewards: WalkingRobotRewardFinetuneCfg = WalkingRobotRewardFinetuneCfg()
    def __post_init__(self):
        super().__post_init__()

        self.events.add_base_mass = None #type: ignore
        self.events.robot_physics_material = None #type: ignore
        self.events.robot_joint_stiffness_and_damping = None #type: ignore
        self.events.robot_center_of_mass = None #type: ignore
        self.events.randomize_actuator_gains = None #type: ignore
 

@configclass
class WalkingRobotPLayEnvCfg(WalkingRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # # self.sim.render_interval = 8
        # self.episode_length_s = 20.0
        # self.sim.dt = 0.001
        # self.decimation = 20
        # self.sim.render_interval = self.decimation

        self.observations.policy.enable_corruption = False

        # self.events.add_base_mass = 
        # self.events.base_com.params = {
        #     "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        #     "com_range": {"x": (-0.2, -0.2), "y": (-0.00, 0.00), "z": (-0.00, 0.00)},
        # }
        self.events.add_base_mass = None #type: ignore
        self.events.robot_physics_material = None #type: ignore
        self.events.robot_joint_stiffness_and_damping = None #type: ignore
        # self.events.robot_center_of_mass = None #type: ignoree
        self.events.reset_robot_joints = None #type: ignore
        self.events.reset_robot_base = None #type: ignore
        self.events.randomize_actuator_gains = None #type: ignore
        self.events.push_robot = None #type: ignore
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

        self.commands.gait_command.ranges.frequencies = (1.0, 1.0)
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

 

@configclass
class WalkingRobotPLayFinetuneEnvCfg(WalkingRobotPLayEnvCfg):
    
    def __post_init__(self):
        super().__post_init__()


    