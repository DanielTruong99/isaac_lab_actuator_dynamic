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
from isaac_lab_actuator_dynamic.assets import LEGWALKING_HIGH_GAIN_AMARTURE_3_CFG, LEGWALKING_HIGH_GAIN_AMARTURE_CFG, LEGWALKING_HIGH_GAIN_AMARTURE_5_CFG
from . import mdp as custom_mdp
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
import torch
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from typing import Optional, Tuple
import torch.nn as nn

@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        filter_shape_paths_expr=None,  # ["/World/ground/terrain/GroundPlane/CollisionPlane"],
        history_length=3,
        track_air_time=True,
    )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = custom_mdp.CustomJointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*"], 
        use_default_offset=True,
        # scale = {
        #     ".*_hip_joint": 0.5,
        #     ".*_hip2_joint": 0.5,
        #     ".*_thigh_joint": 0.5,
        #     ".*_calf_joint": 0.8,
        #     ".*_toe_joint": 0.15,
        # }
        scale=0.25,
    )

@configclass
class ObservationsCfg:
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

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg() 
    critic: CriticCfg = CriticCfg()
    

@configclass 
class EventCfg:

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

    add_link_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_thigh", ".*_hip", ".*_hip2", ".*_calf", ".*_toe"]),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    # randomize_rigid_body_mass_inertia = EventTermCfg(
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

    robot_joint_stiffness_and_damping = EventTermCfg(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.45, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

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
class RewardCfg:
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

    # penalty terms
    # termination_penalty = RewardTermCfg(func=mdp.is_terminated, weight=-100.0)
    dof_torques_l2 = RewardTermCfg(func=mdp.joint_torques_l2, weight=-6.5e-5)
    base_height_l2 = RewardTermCfg(
        func=mdp.base_height_l2,
        weight=-50.0,
        params={"target_height": 0.77},
    )
    dof_vel = RewardTermCfg(
        func=mdp.joint_vel_l2,
        weight=-5.0e-05,
    )
    dof_acc_l2 = RewardTermCfg(func=mdp.joint_acc_l2, weight=-1.0e-6)
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
        params={"min_feet_distance": 0.18,"feet_links_name": ["L_toe", "R_toe"]}
    )
    pen_feet_regulation = RewardTermCfg(
        func=custom_mdp.feet_regulation,
        weight=-0.05, # -0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["L_toe", "R_toe"]),
                "base_height_target": 0.77, "foot_radius": 0.035},
    )

    pen_joint_power_l1 = RewardTermCfg(func=custom_mdp.joint_powers_l1, weight=-2.0e-05)

    pen_foot_landing_vel = RewardTermCfg(
            func=custom_mdp.foot_landing_vel,
            weight=-0.15,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["L_toe", "R_toe"]),
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["L_toe", "R_toe"]),
                    "foot_radius": 0.035, "about_landing_threshold": 0.08},
        )


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
class CommandsCfg:
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
            lin_vel_x=(-0.3, 0.8), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
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

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


@configclass
class WalkingRobotNewtonEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardCfg = RewardCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Simulation settings
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

        # setting plane terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator.curriculum = False #type: ignore
        self.curriculum.terrain_levels = None
        self.episode_length_s = 20.0
        self.sim.dt = 0.005 # 200hz
        self.decimation = 4 # 50hz
        self.sim.render_interval = self.decimation

        # setting robot asset
        self.scene.robot = LEGWALKING_HIGH_GAIN_AMARTURE_3_CFG.replace(prim_path="/World/envs/env_.*/Robot") #type: ignore

@configclass
class RewardFineTuneCfg(RewardCfg):
    action_rate_l2 = RewardTermCfg(
        func=custom_mdp.action_rate_max_kernel, 
        weight=5.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_hip2_joint", ".*_thigh_joint", ".*_toe_joint", ".*_calf_joint"])},
    )
    # action_rate_calf_l2 = RewardTermCfg(
    #     func=custom_mdp.action_rate_max_kernel_per_joint,
    #     weight=2.5,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_calf_joint"])},
    # )

@configclass
class WalkingRobotNewtonEnvFineTuneCfg(WalkingRobotNewtonEnvCfg):
    rewards: RewardFineTuneCfg = RewardFineTuneCfg()

@configclass
class ObservationsPlayCfg(ObservationsCfg):
    """Observation specifications for the play mode."""

    @configclass
    class DebugCfg(ObservationGroupCfg):
        """Observations for debug group."""
        joint_action_L_thigh = ObservationTermCfg(func=custom_mdp.joint_actions, params={"asset_cfg": SceneEntityCfg("robot", joint_names="L_thigh_joint")})
        joint_action_R_thigh = ObservationTermCfg(func=custom_mdp.joint_actions, params={"asset_cfg": SceneEntityCfg("robot", joint_names="R_thigh_joint")})

        joint_action_L_calf = ObservationTermCfg(func=custom_mdp.joint_actions, params={"asset_cfg": SceneEntityCfg("robot", joint_names="L_calf_joint")})
        joint_action_R_calf = ObservationTermCfg(func=custom_mdp.joint_actions, params={"asset_cfg": SceneEntityCfg("robot", joint_names="R_calf_joint")})

        joint_action_L_hip = ObservationTermCfg(func=custom_mdp.joint_actions, params={"asset_cfg": SceneEntityCfg("robot", joint_names="L_hip_joint")})
        joint_action_R_hip = ObservationTermCfg(func=custom_mdp.joint_actions, params={"asset_cfg": SceneEntityCfg("robot", joint_names="R_hip_joint")})

        joint_action_L_hip2 = ObservationTermCfg(func=custom_mdp.joint_actions, params={"asset_cfg": SceneEntityCfg("robot", joint_names="L_hip2_joint")})
        joint_action_R_hip2 = ObservationTermCfg(func=custom_mdp.joint_actions, params={"asset_cfg": SceneEntityCfg("robot", joint_names="R_hip2_joint")})

        desired_contact_states = ObservationTermCfg(func=custom_mdp.desired_contact_states, params={"command_name": "gait_command"})

    debug: DebugCfg = DebugCfg()

@configclass
class WalkingRobotNewtonPlayEnvCfg(WalkingRobotNewtonEnvCfg):

    observations: ObservationsPlayCfg = ObservationsPlayCfg()

    def __post_init__(self):
        super().__post_init__()

        self.events.push_robot = None
        self.events.reset_robot_base = None
        self.events.reset_robot_joints = None
        self.events.robot_joint_stiffness_and_damping = None

        # self.sim.dt = 0.002 # 1000hz
        # self.decimation = 10 # 50hz
        # self.sim.render_interval = self.decimation