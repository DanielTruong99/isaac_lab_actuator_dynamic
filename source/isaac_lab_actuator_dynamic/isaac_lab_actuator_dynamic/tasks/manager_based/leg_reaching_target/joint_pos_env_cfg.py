# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs.mdp.actions import joint_actions


##
# Pre-defined configs
##
from isaac_lab_actuator_dynamic.assets import LEGACTUATORDYNAMIC_CFG, LEGACTUATORDYNAMIC_2_CFG
import isaac_lab_actuator_dynamic.tasks.manager_based.leg_reaching_target.mdp as custom_mdp
##
# Environment configuration
##

class CustomJointPositionAction(joint_actions.JointPositionAction):
    def __init__(self, cfg, env):
        # initialize the action term
        super().__init__(cfg, env)
        
        self.filtered_actions = torch.zeros_like(self.processed_actions)

    def apply_actions(self):
        # set position targets
        self.filtered_actions = 0.999 * self.filtered_actions + (1 - 0.999) * self.processed_actions
        self._asset.set_joint_position_target(self.filtered_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids) -> None:
        self._raw_actions[env_ids] = 0.0
        self.filtered_actions[env_ids] = 0.0

@configclass
class ActionsPlayCfg:
    """Action specifications for the MDP."""

    left_leg_action =  mdp.JointPositionActionCfg(
        class_type=CustomJointPositionAction,
        asset_name="robot", 
        joint_names=["L_hip_joint", "L_hip2_joint", "L_thigh_joint", "L_calf_joint", "L_toe_joint"], 
        scale=1.0, 
        use_default_offset=True
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    left_leg_action =  mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=["L_hip_joint", "L_hip2_joint", "L_thigh_joint", "L_calf_joint", "L_toe_joint"], 
        scale=1.0, 
        use_default_offset=True
    )


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.7, 0.7),
            "velocity_range": (-1.0, 1.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    # end_effector_position_tracking = RewTerm(
    #     func=mdp.position_command_error,
    #     weight=-0.2,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names="ee_link"), "command_name": "ee_pose"},
    # )
    end_effector_position_tracking_fine_grained = RewTerm(
        func=custom_mdp.position_command_error_multi_scale,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="L_toe"), "std": 0.25, "command_name": "L_toe"},
    )
    end_effector_orientation_tracking = RewTerm(
        func=custom_mdp.orientation_error_tanh,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="L_toe"),  "std": 1.2, "command_name": "L_toe"},
    )
    orientation_bonus = RewTerm(
        func=custom_mdp.position_orientation_bonus,
        weight=100.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="L_toe"), "command_name": "L_toe"},
    )

    # penalties
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-5e-3)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_torque = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-7e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    terminated = RewTerm(func=mdp.is_terminated, weight=-100.0)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    far_from_target = DoneTerm(func=custom_mdp.far_from_target, params={"asset_cfg": SceneEntityCfg("robot", body_names=["L_toe"]), "command_name": "L_toe"})

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    L_toe = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="L_toe",
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.1, 0.35),
            pos_y=(-0.1, 0.2),
            pos_z=(-0.78, -0.41),
            roll=(0.0, 0.0),
            pitch=(-math.pi * 10.0/ 180.0, 0.0),
            yaw=(-math.pi * 10.0/ 180.0, math.pi * 10.0/ 180.0),
        ),
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, 
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["L_hip_joint", "L_hip2_joint", "L_thigh_joint", "L_calf_joint", "L_toe_joint"])}, 
            noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, 
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["L_hip_joint", "L_hip2_joint", "L_thigh_joint", "L_calf_joint", "L_toe_joint"])}, 
            noise=Unoise(n_min=-0.15, n_max=0.15)
        )
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "L_toe"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class DebugCfg(ObsGroup):
        """Observations for debug group."""

        # observation terms (order preserved)
        L_toe_v = ObsTerm(func=custom_mdp.body_lin_vel, params={"asset_cfg": SceneEntityCfg("robot", body_names="L_toe")})
        L_toe_w = ObsTerm(
            func=custom_mdp.body_ang_vel, params={"asset_cfg": SceneEntityCfg("robot", body_names="L_toe")}
        )
        torque = ObsTerm(func=mdp.joint_effort, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["L_hip_joint", "L_hip2_joint", "L_thigh_joint", "L_calf_joint", "L_toe_joint"])})

        def __post_init__(self):
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    debug: DebugCfg = DebugCfg()



@configclass
class LeftLegReachEnvCfg(ReachEnvCfg):

    rewards: RewardsCfg = RewardsCfg()
    event: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # set the simulation parameters
        self.decimation = 10
        self.sim.dt = 0.001
        self.episode_length_s = 5.0

        # switch robot to leg actuator dynamic 2
        self.scene.robot = LEGACTUATORDYNAMIC_2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.table = None 
        self.scene.ground = None
        self.scene.light = None

        # disable the curriculum
        self.curriculum.action_rate = None
        self.curriculum.joint_vel = None


@configclass
class LeftLegReachEnvCfg_PLAY(LeftLegReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 3.0

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        # disable randomization for play
        self.events.reset_robot_joints.params = {"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)}

        # enable debug visualization for ee pose command
        self.commands.L_toe.debug_vis = True

        # change action to play version
        self.actions = ActionsPlayCfg()
