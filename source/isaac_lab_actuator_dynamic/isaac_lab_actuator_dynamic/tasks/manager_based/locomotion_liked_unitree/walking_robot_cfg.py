import math
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import torch

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
    EventCfg,
    ObservationsCfg,
)       
from isaaclab.managers import TerminationTermCfg as DoneTerm

from isaaclab.utils import configclass
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    EventTermCfg,
)
from isaaclab.utils.noise import AdditiveUniformNoiseCfg, AdditiveGaussianNoiseCfg

##
# User defined configs
##
from isaac_lab_actuator_dynamic.assets import LEGACTUATORDYNAMIC_CFG, LEGACTUATORDYNAMIC_2_CFG, LEGWALKING_CFG
from . import mdp as custom_mdp

@configclass
class WalkingRobotObservationsCfg(ObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""
        # base state
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.05), clip=(-100.0, 100.0), scale=0.25)
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.025), clip=(-100.0, 100.0), scale=1.0)
        
        # velocity command
        velocity_commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # joint state
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=1.0)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel, noise=AdditiveGaussianNoiseCfg(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=0.05)
        
        # last action
        actions = ObservationTermCfg(func=mdp.last_action)

        # gaits
        phase = ObservationTermCfg(func=custom_mdp.get_phase)

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
        heights = ObservationTermCfg(func=mdp.height_scan,params={"sensor_cfg": SceneEntityCfg("height_scanner")})
        robot_base_pos = ObservationTermCfg(func=mdp.root_pos_w)
        robot_base_quat = ObservationTermCfg(func=mdp.root_quat_w)
        robot_base_lin_vel = ObservationTermCfg(func=mdp.root_lin_vel_w)
        robot_base_ang_vel = ObservationTermCfg(func=mdp.root_ang_vel_w)

        # velocity command
        velocity_commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # joint state
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel)

        # last action
        last_action = ObservationTermCfg(func=mdp.last_action)

        # contact state and phase
        contact_state = ObservationTermCfg(
            func=custom_mdp.contact_state, 
            params={
                "sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=["L_toe", "R_toe"]),
                "asset_cfg": SceneEntityCfg("robot", body_names=["L_toe", "R_toe"]),
            }
        )
        phase = ObservationTermCfg(func=custom_mdp.get_phase)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class DebugCfg(ObservationGroupCfg):
        """Observations for debug group."""
        joint_torque = ObservationTermCfg(func=custom_mdp.joint_torque)
        base_height = ObservationTermCfg(func=mdp.base_pos_z)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg() 
    critic: CriticCfg = CriticCfg()
    #! Just for debugging
    debug: DebugCfg = DebugCfg()
    

@configclass 
class WalkingRobotEventCfg(EventCfg):
    update_phase = EventTermCfg(
        func=custom_mdp.update_phase,
        mode="interval",
        interval_range_s=(0.0, 0.0),
    )

    def __post_init__(self):
        super().__post_init__() #type: ignore

        ''' #!Domain randomization setup
            The default domain randomization setup includes:
            1. physic material
            2. add base mass
            3. base external force torque
            4. reset base
            5. reset robot joint
            6. push robot

        '''
        # self.physics_material.params["dynamic_friction_range"] = [0.1, 1.25]
        self.physics_material = None
        # self.add_base_mass.params["mass_distribution_params"] = [-1.0, 3.0]
        self.add_base_mass = None
        self.push_robot.params = {
            "velocity_range": {
                "x": [-1.5, 1.5],
                "y": [-1.5, 1.5],
            }
        }
        self.push_robot.interval_range_s = (2.5, 2.5)
        self.base_external_force_torque = None 
        self.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        # # self.reset_robot_joints.func = mdp.reset_joints_by_offset
        self.reset_robot_joints.params = {
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        }

@configclass
class WalkingRobotRewardCfg:
    track_lin_vel_xy_exp = RewardTermCfg(
        func=mdp.track_lin_vel_xy_exp, weight=3.0, params={"command_name": "base_velocity", "std": math.sqrt(0.2)}
    )
    track_ang_vel_z_exp = RewardTermCfg(
        func=mdp.track_ang_vel_z_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.2)}
    )
    is_alive = RewardTermCfg(
        func=mdp.is_alive,
        weight=1.0,
    )
    feet_schedule_contact = RewardTermCfg(
        func=custom_mdp.feet_schedule_contact_with_cmd,
        weight=1.2,
        params={"sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=["L_toe", "R_toe"])},
    )

    # penalty terms
    dof_torques_l2 = RewardTermCfg(func=custom_mdp.weighted_joint_torques_l2, weight=-8.0e-5)
    base_height_l2 = RewardTermCfg(
        func=mdp.base_height_l2,
        weight=-20.0,
        params={"target_height": 0.78},
    )
    dof_vel = RewardTermCfg(
        func=mdp.joint_vel_l2,
        weight=-1e-3,
    )
    dof_acc_l2 = RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7)
    lin_vel_z_l2 = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05)
    action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.03)
    dof_pos_limits = RewardTermCfg(func=mdp.joint_pos_limits, weight=-2.0)
    undesired_contacts = RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_hip", ".*_hip2", "base"]), "threshold": 10.0},
    )
    action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.4)
    flat_orientation_l2 = RewardTermCfg(func=mdp.flat_orientation_l2, weight=-10.0)
    action_norm = RewardTermCfg(
        func=mdp.action_l2,
        weight=-0.001,
    )
    joint_deviation = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_hip2_joint"])},
    )
    # feet_distance = RewardTermCfg(
    #     func=custom_mdp.feet_distance,
    #     weight=-100,
    #     params={"min_feet_distance": 0.115,"feet_links_name": ["foot_[RL]_Link"]}
    # )

    feet_height = RewardTermCfg(
        func=custom_mdp.feet_height,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=["L_toe", "R_toe"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["L_toe", "R_toe"]),
        },
    )

    stand_still_contact = RewardTermCfg(
        func=custom_mdp.stand_still_contact,
        weight=-0.7,
        params={
            "sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=["L_toe", "R_toe"]),
        },
    )

    stand_still = RewardTermCfg(
        func=custom_mdp.stand_still,
        weight=-0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot")
        },
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
class WalkingRobotCommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        class_type=CustomUniformVelocityCommand,
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.35,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.75, 3.5), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-0.5, 0.5)
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
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"asset_cfg": SceneEntityCfg("robot"), "minimum_height": 0.3},)

@configclass
class WalkingRobotEnvCfg(LocomotionVelocityRoughEnvCfg):
    observations: WalkingRobotObservationsCfg = WalkingRobotObservationsCfg()
    rewards: WalkingRobotRewardCfg = WalkingRobotRewardCfg()
    events: WalkingRobotEventCfg = WalkingRobotEventCfg()
    commands: WalkingRobotCommandsCfg = WalkingRobotCommandsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        ''' #!Terrain setup'''
        # self.scene.terrain.terrain_generator = custom_mdp.TERRAINS_CFG
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator.curriculum = False #type: ignore
        self.curriculum.terrain_levels = None
        self.sim.episode_length_s = 10.0

        ''' #!Action setup
            The default action space setup includes:
            1. joint position command
        '''
        # q_cmd = q_default + scale * network_output
        self.actions.joint_pos.scale = 1.0
        
        self.scene.robot = LEGWALKING_CFG.replace(prim_path="/World/envs/env_.*/Robot") #type: ignore 
        # self.scene.height_scanner = None #type: ignore

        # self.sim.dt = 0.001
        # self.decimation = 10
        step_dt = self.sim.dt * self.decimation
        self.events.update_phase.interval_range_s = (step_dt, step_dt)

        ''' #!Termination setup
            The default termination setup includes:
            1. base contact
            2. time out
        '''
        #* Remain the default termination setup

@configclass
class WalkingRobotEnvPLayCfg(WalkingRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.enable_corruption = False

        self.events.add_base_mass = None #type: ignore
        self.events.base_external_force_torque = None
        self.events.push_robot = None #type: ignore
        self.events.reset_base = None #type: ignore
        self.events.reset_robot_joints = None #type: ignore

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator.curriculum = False #type: ignore
        self.curriculum.terrain_levels = None #type: ignore

        self.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
   

        # self.viewer.asset_name = "robot"
        # self.viewer.origin_type = "asset_root"
        # self.viewer.eye = (0.0, 5.0, 2.0)


        # self.sim.use_fabric = False
        # self.sim.device = "cpu"
        # self.actions.joint_pos.scale = {"R_toe_joint": 0.001}

 


    



    