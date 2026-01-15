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
from isaac_lab_actuator_dynamic.assets import LEGWALKING_5_CFG, LEGWALKING_HIGH_GAIN_AMARTURE_CFG, LEGWALKING_HIGH_GAIN_AMARTURE_5_CFG, ACTION_SCALE
from . import mdp as custom_mdp
from . import dm_mdp as dm_mdp
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
import torch
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from typing import Optional, Tuple
import torch.nn as nn

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


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
        force_threshold=10.0
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
class RealActionsCfg:
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
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(
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
        # command = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "motion"})
        # motion_body_pos_b = ObservationTermCfg(
        #     func=dm_mdp.motion_body_pos_b, params={"command_name": "motion"}, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.15)
        # )
        # motion_body_ori_b = ObservationTermCfg(
        #     func=dm_mdp.motion_body_ori_b, params={"command_name": "motion"}, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.03)
        # )
        

        # # base state
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.2), clip=(-100.0, 100.0), scale=1.0)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.12), clip=(-100.0, 100.0), scale=1.0)
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.025), clip=(-100.0, 100.0), scale=1.0)
        
        # joint state
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=1.0)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.35), clip=(-100.0, 100.0), scale=1.0)
        
        # last action
        last_action = ObservationTermCfg(func=mdp.last_action, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=1.0)

        # vel cmds
        velocity_commands_b = ObservationTermCfg(func=dm_mdp.generated_commands_b, params={"command_name": "base_velocity"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True


    @configclass
    class CriticCfg(ObservationGroupCfg):
        """Observations for policy group."""
        # command = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "motion"})
        # motion_anchor_pos_b = ObservationTermCfg(func=dm_mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        # motion_anchor_ori_b = ObservationTermCfg(func=dm_mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        # body_pos = ObservationTermCfg(func=dm_mdp.robot_body_pos_b, params={"command_name": "motion"})
        # body_ori = ObservationTermCfg(func=dm_mdp.robot_body_ori_b, params={"command_name": "motion"})
        # motion_body_pos_b = ObservationTermCfg(func=dm_mdp.motion_body_pos_b, params={"command_name": "motion"})
        # motion_body_ori_b = ObservationTermCfg(func=dm_mdp.motion_body_ori_b, params={"command_name": "motion"})

        # base state
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel)
        proj_gravity = ObservationTermCfg(func=mdp.projected_gravity)
        body_lin_vel_w = ObservationTermCfg(func=dm_mdp.body_lin_vel_w)

        # joint state
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel)

        # last action
        last_action = ObservationTermCfg(func=mdp.last_action)

        # vel cmds
        velocity_commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})

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
            "stiffness_distribution_params": (0.85, 1.105),
            "damping_distribution_params": (1.0, 2.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # reset
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

    reset_robot_state = EventTermCfg(
        func=dm_mdp.reset_robot_state_by_motion,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.01, 0.01),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.2, 0.2),
            },
            "velocity_range": VELOCITY_RANGE,
            "joint_position_range": (-0.1, 0.1),
        },
    )


@configclass
class RewardCfg:
    # motion_global_anchor_pos = RewardTermCfg(
    #     func=dm_mdp.motion_global_anchor_position_error_exp,
    #     weight=0.5,
    #     params={"command_name": "motion", "std": 0.3},
    # )
    # motion_global_anchor_ori = RewardTermCfg(
    #     func=dm_mdp.motion_global_anchor_orientation_error_exp,
    #     weight=0.1,
    #     params={"command_name": "motion", "std": 0.4},
    # )
    motion_body_pos = RewardTermCfg(
        func=dm_mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewardTermCfg(
        func=dm_mdp.motion_relative_body_orientation_error_exp,
        weight=1.5,
        params={"command_name": "motion", "std": 0.4},
    )
    # motion_body_lin_vel = RewardTermCfg(
    #     func=dm_mdp.motion_global_body_linear_velocity_error_exp,
    #     weight=1.0,
    #     params={"command_name": "motion", "std": 1.0},
    # )
    # motion_body_ang_vel = RewardTermCfg(
    #     func=dm_mdp.motion_global_body_angular_velocity_error_exp,
    #     weight=1.0,
    #     params={"command_name": "motion", "std": 3.14},
    # )

    # motion_joint_pos = RewardTermCfg(
    #     func=dm_mdp.motion_joint_pos_error_exp,
    #     weight=0.5,
    #     params={"command_name": "motion", "std": 0.15},
    # )
    # motion_joint_vel = RewardTermCfg(
    #     func=dm_mdp.motion_joint_vel_error_exp,
    #     weight=0.1 * 0.5,
    #     params={"command_name": "motion", "std": 0.5},
    # )

    track_lin_vel_xy_exp = RewardTermCfg(
        func=dm_mdp.track_lin_vel_xy_exp_max, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # track_lin_vel_xy_d = RewardTermCfg(
    #     func=dm_mdp.track_lin_vel_xy_exp_d, weight=1.0 * 0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    # )

    # track_ang_vel_z_exp = RewardTermCfg(
    #     func=mdp.track_ang_vel_z_exp, weight=0.2 * 0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    # )

    # penalty terms
    action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.3)
    pen_action_smoothness = RewardTermCfg(func=custom_mdp.ActionSmoothnessPenalty, weight=-0.2)
    # # dof_pos_limits = RewardTermCfg(func=mdp.joint_pos_limits, weight=-0.1)
    # undesired_contacts = RewardTermCfg(
    #     func=mdp.undesired_contacts,
    #     weight=-0.5,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_hip", ".*_hip2", "base"]), "threshold": 10.0},
    # )


class CustomUniformVelocityCommand(mdp.UniformVelocityCommand):
    """Custom uniform velocity command generator configuration."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self.metrics["num_standing_envs"] = torch.zeros(self.num_envs, device=self.device)

    # def _resample_command(self, env_ids):
    #     super()._resample_command(env_ids)
    #     motion_command = self._env.command_manager.get_term("motion")
    #     standing_ids_motion = motion_command.motion.motion_ids == 0
    #     self.is_standing_env[:] = False
    #     self.is_standing_env[standing_ids_motion] = True
        

    def _update_metrics(self):
        super()._update_metrics()
        # -- metrics
        self.metrics["num_standing_envs"] = self.is_standing_env.float()

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        magnitude, direction = torch.split(self.command, [1, 2], dim=-1)
        vel_cmds = magnitude * direction
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(vel_cmds)
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_w[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)


MOTION_FOLDER = "source/isaac_lab_actuator_dynamic/isaac_lab_actuator_dynamic/tasks/manager_based/sim2real_dm_locomotion/motions/sub_motions/"
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    motion = dm_mdp.MotionCommandCfg(
        asset_name="robot",
        motion_file=[   
            # MOTION_FOLDER + "stand.npz",
            MOTION_FOLDER + "walk_around.npz",
            MOTION_FOLDER + "walk_circle.npz", 
            MOTION_FOLDER + "walk_straight_2.npz",
            MOTION_FOLDER + "walk_straight_3.npz",
            MOTION_FOLDER + "walk_straight_4.npz",
            MOTION_FOLDER + "walk_straight.npz",
            
        ],
        anchor_body_name="base",
        body_names=[
            "base",
            "L_hip",
            "L_hip2",
            "L_thigh",
            "L_calf",
            "L_toe",
            "R_hip",
            "R_hip2",
            "R_thigh",
            "R_calf",
            "R_toe",
        ],
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
    )

    # ! lin_vel_x: magnitude
    # ! lin_vel_y, ang_vel_z: direction
    # ! global frame
    base_velocity = mdp.UniformVelocityCommandCfg(
        class_type=CustomUniformVelocityCommand,
        asset_name="robot",
        resampling_time_range=(3.0, 5.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=1.0,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.5), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
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
class DMWalkingRobotNewtonEnvCfg(ManagerBasedRLEnvCfg):
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
        self.episode_length_s = 10.0
        self.sim.dt = 0.005 # 200hz
        self.decimation = 4 # 0hz
        self.sim.render_interval = self.decimation

        # setting robot asset
        self.scene.robot = LEGWALKING_5_CFG.replace(prim_path="/World/envs/env_.*/Robot") #type: ignore
        self.actions.joint_pos.scale = ACTION_SCALE


# @configclass
# class ObservationsPlayCfg(ObservationsCfg):
#     """Observation specifications for the play mode."""

#     @configclass
#     class DebugCfg(ObservationGroupCfg):
#         """Observations for debug group."""
#         joint_action_L_thigh = ObservationTermCfg(func=custom_mdp.joint_actions, params={"asset_cfg": SceneEntityCfg("robot", joint_names="L_thigh_joint")})
#         joint_action_R_thigh = ObservationTermCfg(func=custom_mdp.joint_actions, params={"asset_cfg": SceneEntityCfg("robot", joint_names="R_thigh_joint")})

#         joint_action_L_calf = ObservationTermCfg(func=custom_mdp.joint_actions, params={"asset_cfg": SceneEntityCfg("robot", joint_names="L_calf_joint")})
#         joint_action_R_calf = ObservationTermCfg(func=custom_mdp.joint_actions, params={"asset_cfg": SceneEntityCfg("robot", joint_names="R_calf_joint")})

#         joint_action_L_hip = ObservationTermCfg(func=custom_mdp.joint_actions, params={"asset_cfg": SceneEntityCfg("robot", joint_names="L_hip_joint")})
#         joint_action_R_hip = ObservationTermCfg(func=custom_mdp.joint_actions, params={"asset_cfg": SceneEntityCfg("robot", joint_names="R_hip_joint")})

#         joint_action_L_hip2 = ObservationTermCfg(func=custom_mdp.joint_actions, params={"asset_cfg": SceneEntityCfg("robot", joint_names="L_hip2_joint")})
#         joint_action_R_hip2 = ObservationTermCfg(func=custom_mdp.joint_actions, params={"asset_cfg": SceneEntityCfg("robot", joint_names="R_hip2_joint")})

#         desired_contact_states = ObservationTermCfg(func=custom_mdp.desired_contact_states, params={"command_name": "gait_command"})

#     debug: DebugCfg = DebugCfg()

@configclass
class DMWalkingRobotNewtonPlayEnvCfg(DMWalkingRobotNewtonEnvCfg):

    # observations: ObservationsPlayCfg = ObservationsPlayCfg()

    def __post_init__(self):
        super().__post_init__()

        self.events.push_robot = None
        self.events.reset_robot_base = None
        self.events.reset_robot_joints = None
        self.events.robot_joint_stiffness_and_damping = None

        self.events.reset_robot_state = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        # self.sim.dt = 0.002 # 1000hz
        # self.decimation = 10 # 50hz
        # self.sim.render_interval = self.decimation


@configclass
class StudentPolicyCfg(ObservationGroupCfg):
    """Observations for policy group."""

    # observation terms (order preserved)
    base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.05), scale=0.25)
    projected_gravity = ObservationTermCfg(
        func=mdp.projected_gravity,
        noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.025),
    )
    joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01))
    joint_vel = ObservationTermCfg(func=mdp.joint_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), scale=0.05)
    actions = ObservationTermCfg(func=mdp.last_action, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01))

    # vel cmds
    velocity_commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class TeacherStudentObservationsCfg:
    """Observation specifications for the MDP."""

    # the policy is the student policy
    policy: StudentPolicyCfg = StudentPolicyCfg()
    # the teacher gets is the privileged observations
    teacher = ObservationsCfg.PolicyCfg()

@configclass
class DMWalkingRobotNewtonEnvCfg_TeacherStudentEnvCfg(DMWalkingRobotNewtonEnvCfg):
    observations: TeacherStudentObservationsCfg = TeacherStudentObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

@configclass
class StudentObservationsCfg:
    """Observation specifications for the MDP."""

    # the policy is the student policy
    policy: StudentPolicyCfg = StudentPolicyCfg()


@configclass
class DMWalkingRobotNewtonEnvCfg_StudentEnvCfg(DMWalkingRobotNewtonEnvCfg):
    observations: StudentObservationsCfg = StudentObservationsCfg()
    actions: RealActionsCfg = RealActionsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()