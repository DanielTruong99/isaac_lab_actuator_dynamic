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

from isaaclab.utils import configclass
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    EventTermCfg,
)
from isaaclab.utils.noise import AdditiveUniformNoiseCfg

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
        # base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, noise=AdditiveUniformNoiseCfg(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=AdditiveUniformNoiseCfg(n_min=-0.2, n_max=0.2))
        projected_gravity = ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=AdditiveUniformNoiseCfg(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=AdditiveUniformNoiseCfg(n_min=-0.01, n_max=0.01))
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, scale=0.15, noise=AdditiveUniformNoiseCfg(n_min=-1.5, n_max=1.5))
        actions = ObservationTermCfg(func=mdp.last_action)
        phase = ObservationTermCfg(func=custom_mdp.get_phase)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True


    @configclass
    class CriticCfg(ObservationGroupCfg):
        """Observations for policy group."""
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, noise=AdditiveUniformNoiseCfg(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=AdditiveUniformNoiseCfg(n_min=-0.2, n_max=0.2))
        projected_gravity = ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=AdditiveUniformNoiseCfg(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=AdditiveUniformNoiseCfg(n_min=-0.01, n_max=0.01))
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, scale=0.15, noise=AdditiveUniformNoiseCfg(n_min=-1.5, n_max=1.5))
        actions = ObservationTermCfg(func=mdp.last_action)
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

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg() 
    critic: CriticCfg = CriticCfg()
    #! Just for debugging
    # debug: DebugCfg = DebugCfg()
    

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
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.5, 0.5),
            "roll": (-0.5, 0.5),
            "pitch": (-0.5, 0.5),
            "yaw": (-0.5, 0.5),
            },
        }
        self.reset_robot_joints.func = mdp.reset_joints_by_offset
        self.reset_robot_joints.params = {
            "position_range": (-0.0, 0.0),
            "velocity_range": (-0.0, 0.0),
        }

@configclass
class WalkingRobotRewardCfg(RewardsCfg):
    # track_lin_vel_xy_exp = RewardTermCfg(
    #     func=custom_mdp.weighted_track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    # )
    # track_ang_vel_z_exp = RewardTermCfg(
    #     func=custom_mdp.weighted_track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    # )

    track_lin_vel_xy_exp = RewardTermCfg(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewardTermCfg(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    dof_torques_l2 = RewardTermCfg(func=custom_mdp.weighted_joint_torques_l2, weight=-1.0e-5)


    base_height_l2 = RewardTermCfg(
        func=mdp.base_height_l2,
        weight=-0.5,
        params={"target_height": 0.78},
    )

    dof_vel = RewardTermCfg(
        func=mdp.joint_vel_l2,
        weight=-1e-3,
    )

    joint_deviation_hip = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_hip2_joint"])},
    )

    is_alive = RewardTermCfg(
        func=custom_mdp.weighted_is_alive,
        weight=0.15,
    )

    is_terminated = RewardTermCfg(
        func=mdp.is_terminated,
        weight=-100.0,
    )

    feet_schedule_contact = RewardTermCfg(
        func=custom_mdp.feet_schedule_contact_with_cmd,
        weight=1.0,
        params={"sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=["L_toe", "R_toe"])},
    )

    feet_height = RewardTermCfg(
        func=custom_mdp.feet_height,
        weight=-0.7,
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
        rel_standing_envs=0.3,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.0, 1.5), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.2, 0.2)
        ),
    )

@configclass
class WalkingRobotEnvCfg(LocomotionVelocityRoughEnvCfg):
    observations: WalkingRobotObservationsCfg = WalkingRobotObservationsCfg()
    rewards: WalkingRobotRewardCfg = WalkingRobotRewardCfg()
    events: WalkingRobotEventCfg = WalkingRobotEventCfg()
    commands: WalkingRobotCommandsCfg = WalkingRobotCommandsCfg()

    def __post_init__(self):
        super().__post_init__()

        ''' #!Terrain setup'''
        # self.scene.terrain.terrain_generator = custom_mdp.TERRAINS_CFG
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator.curriculum = False #type: ignore
        self.curriculum.terrain_levels = None
        self.sim.episode_length_s = 5.0

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

        ''' #!Reward setup'''
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 2.5
        self.rewards.dof_pos_limits.weight = -1.5
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.feet_air_time = None #type: ignore
        self.rewards.undesired_contacts = None #type: ignore
        # self.rewards.feet_height = None #type: ignore

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

        # self.viewer.asset_name = "robot"
        # self.viewer.origin_type = "asset_root"
        # self.viewer.eye = (0.0, 5.0, 2.0)


        # self.sim.use_fabric = False
        # self.sim.device = "cpu"

 


    



    