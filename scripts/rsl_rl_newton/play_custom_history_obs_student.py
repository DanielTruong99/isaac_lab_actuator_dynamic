# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--newton_visualizer", action="store_true", default=False, help="Enable Newton rendering.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import matplotlib.pyplot as plt

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.utils.timer import Timer

Timer.enable = False
Timer.enable_display_output = False

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.managers.action_manager import ActionManager
from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaac_lab_actuator_dynamic.assets import FLATFOOT_WITHPIN_HIGHGAIN_CFG, HIGHGAIN_ACTION_SCALE
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.utils.buffers import CircularBuffer

# PLACEHOLDER: Extension template (do not remove this comment)
import isaac_lab_actuator_dynamic.tasks  # noqa: F401

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.enable_newton_rendering = args_cli.newton_visualizer

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # Set play mode for Newton viewer if using Newton visualizer
    if args_cli.newton_visualizer:
        # Set visualizer to play mode in Newton config
        if hasattr(env_cfg.sim, "newton_cfg"):
            env_cfg.sim.newton_cfg.visualizer_train_mode = False
        else:
            # Create newton_cfg if it doesn't exist
            from isaaclab.sim._impl.newton_manager_cfg import NewtonCfg

            newton_cfg = NewtonCfg()
            newton_cfg.visualizer_train_mode = False
            env_cfg.sim.newton_cfg = newton_cfg

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # TODO reset environment
    # TODO reset joint states to initial configuration
    num_envs = env.unwrapped.num_envs
    asset: Articulation = env.unwrapped.scene["robot"]
    action_manager: ActionManager = env.unwrapped.action_manager
    joint_pos_action: JointPositionAction = action_manager._terms["joint_pos"]

    joint_pos = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=env.unwrapped.device)
    joint_pos = joint_pos.repeat(num_envs, 1)
    joint_vel = torch.zeros(num_envs, asset.num_joints, device=env.unwrapped.device)
    asset.write_joint_state_to_sim(
        joint_pos,
        joint_vel,
    )
    # TODO reset robot base to initial configuration
    positions = torch.tensor([0.0, 0.0, 0.645], device=env.unwrapped.device)
    positions = positions.repeat(num_envs, 1)
    orientations = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.unwrapped.device)
    orientations = orientations.repeat(num_envs, 1)
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1))

    # TODO data collection setup
    collect_time = 10.0  # seconds
    max_num_steps = int(collect_time / dt)
    joint_vels = torch.zeros((max_num_steps, num_envs, asset.num_joints), device=env.unwrapped.device)


    obs = env.get_observations()
    timestep = 0
    counter = 0
    policy_counter = 0
    state = "STAND"
    pre_state = "NONE"
    signal = "entry"
    prev_applied_torque = torch.zeros_like(asset.data.applied_torque, device=env.unwrapped.device)
    circular_buffer_list = []
    data_list = [None] * 8
    for _ in range(8):
        circular_buffer_list.append(CircularBuffer(max_len=10, batch_size=num_envs, device=env.unwrapped.device))

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        signal = "50hz_timeout"
        
        if state != pre_state:
            print(f"State changed from {pre_state} to {state}")
            pre_state = state
            signal = "entry"

        if state == "STAND":
            if signal == "entry":
                # stiffness = [50.0, 70.0, 700.0, 700.0, 50.0, 50.0, 70.0, 700.0, 700.0, 50.0]
                # stiffness = torch.tensor(stiffness, device=env.unwrapped.device)
                # stiffness.repeat(num_envs, 1)
                # dampings = [ 3.0,  4.0,  4.0,  1.0, 1.0, 3.0,  4.0,  4.0,  1.0,  1.0]
                # dampings = torch.tensor(dampings, device=env.unwrapped.device)
                # dampings = dampings.repeat(num_envs, 1)
                # asset.write_joint_stiffness_to_sim(stiffness)
                # asset.write_joint_damping_to_sim(dampings)

                # modify joint position action settings, the actions will directly set the joint positions
                joint_pos_action._offset = torch.zeros((num_envs,asset.num_joints), device=env.unwrapped.device)
                joint_pos_action._scale = 1.0

                actions = torch.zeros((num_envs, asset.num_joints), device=env.unwrapped.device)
                max_time = 0.2; max_counter = max_time / dt
                first_pos_cmd = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=env.unwrapped.device)
                first_pos_cmd = first_pos_cmd.repeat(num_envs, 1)
                first_joint_pos = asset.data.joint_pos.clone()
            
            if signal == "50hz_timeout":
                alpha = counter / max_counter
                actions = (1-alpha) * first_joint_pos + alpha * first_pos_cmd

                if alpha >= 1.0:
                    state = "WALK"
                    continue
                
        if state == "WALK":
            if signal == "entry":
                # stiffness = [45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0]
                # stiffness = torch.tensor(stiffness, device=env.unwrapped.device)
                # stiffness.repeat(num_envs, 1)
                # dampings = [ 1.5,  1.5,  1.5,  1.5, 0.8, 1.5,  1.5,  1.5,  1.5, 0.8]
                # dampings = torch.tensor(dampings, device=env.unwrapped.device)
                # dampings = dampings.repeat(num_envs, 1)
                # asset.write_joint_stiffness_to_sim(stiffness)
                # asset.write_joint_damping_to_sim(dampings)

                # modify joint position action settings, for policy output scaling
                action_scales = [value for _, value in HIGHGAIN_ACTION_SCALE.items()]
                joint_pos_action._offset = torch.tensor([0.0, 0.0, 0.65, -1.05, 0.4, 0.0, 0.0, 0.65, -1.05, 0.4], device=env.unwrapped.device)
                joint_pos_action._scale = torch.tensor(action_scales, device=env.unwrapped.device)

                # standing command
                vel_cmds = torch.zeros((num_envs, 3), device=env.unwrapped.device)
                gait_cmds = torch.tensor([0.0, 0.0, 1.0, 0.0], device=env.unwrapped.device) # freq, offset, duration
                gait_cmds = gait_cmds.repeat(num_envs, 1)
                counter_5s = 3.0 / dt
                env.unwrapped.action_manager._terms["joint_pos"].alpha = 0.45  # fc = 3.5 hz
                # counter = 0
                policy_counter = 0
                filtered_vel_cmds = torch.zeros((num_envs, 3), device=env.unwrapped.device)

            if signal == "50hz_timeout":
                if counter > counter_5s:
                    robot_orientation = asset.data.root_com_quat_w.clone()
                    yaw = torch.atan2(2.0 * (robot_orientation[:, 0] * robot_orientation[:, 3] + robot_orientation[:, 1] * robot_orientation[:, 2]),
                                     1.0 - 2.0 * (robot_orientation[:, 2]**2 + robot_orientation[:, 3]**2))
                    wz_cmd = -1.5 * yaw  # P controller to face forward
                    wz_cmd = torch.clamp(wz_cmd, min=-0.8, max=0.8)
                    vel_cmds = torch.tensor([0.35, 0.0, wz_cmd], device=env.unwrapped.device)
                    alpha = 0.9
                    filtered_vel_cmds = alpha * filtered_vel_cmds + (1 - alpha) * vel_cmds
                    f = 1.2*vel_cmds[0]/0.45
                    f = torch.clamp(f, min=0.8, max=1.6)
                    gait_cmds = torch.tensor([1.2, 0.5, 0.5, 0.1], device=env.unwrapped.device) # freq, offset, duration
                    env.unwrapped.action_manager._terms["joint_pos"].alpha = 0.45  # fc = 3.5 hz
                    gait_cmds = gait_cmds.repeat(num_envs, 1)
                    vel_cmds = vel_cmds.repeat(num_envs, 1)





                policy_counter += 1

                # Calculate gait indices based on episode length
                # gait_indices = torch.remainder(policy_counter * env.unwrapped.step_dt * gait_cmds[:, 0], 1.0)
                # gait_indices = gait_indices.unsqueeze(-1)
                # sin_phase = torch.sin(2 * torch.pi * gait_indices)
                # cos_phase = torch.cos(2 * torch.pi * gait_indices)
                # gait_phase = torch.cat([sin_phase, cos_phase], dim=-1)
                    
                
                with torch.inference_mode():
                    if counter < max_num_steps - 1:
                        # action = env.unwrapped.action_manager.action.clone()
                        # prev_action = env.unwrapped.action_manager.prev_action.clone()
                        # filter_actions = env.unwrapped.action_manager._terms["joint_pos"].filtered_actions.clone()
                        # prev_filtered_actions = env.unwrapped.action_manager._terms["joint_pos"].prev_filtered_actions.clone()
                        # # joint_vels[counter] = (filter_actions - prev_filtered_actions)
                        applied_torque = asset.data.applied_torque.clone()
                        joint_vels[counter] = (applied_torque - prev_applied_torque)
                        prev_applied_torque = applied_torque
                        # joint_pos = asset.data.joint_pos.clone()
                        # joint_vels[counter] = joint_pos
                    else:
                        joint_vels = joint_vels.cpu().numpy()
                        fig, axes = plt.subplots(2, 5, figsize=(8, 16))
                        joint_names = asset.joint_names
                        for j in range(asset.num_joints):
                            axes[j//5, j%5].plot(joint_vels[:, 0 , j], label=f'Joint {j}')
                            axes[j//5, j%5].set_xlabel('Time step')
                            axes[j//5, j%5].set_ylabel('Joint Velocity (rad/s)')
                            axes[j//5, j%5].set_title(f'Joint {joint_names[j]} Velocity over Time')
                            axes[j//5, j%5].legend()
                            axes[j//5, j%5].grid()
                        plt.tight_layout()
                        plt.show()
                    ang_vel = obs["play_obs"][:, :3]
                    circular_buffer_list[0].append(ang_vel)
                    data_list[0] = circular_buffer_list[0].buffer.reshape(num_envs, -1)

                    projected_g = obs["play_obs"][:, 3:6]
                    circular_buffer_list[1].append(projected_g)
                    data_list[1] = circular_buffer_list[1].buffer.reshape(num_envs, -1)

                    joint_pos = obs["play_obs"][:, 6:16]
                    circular_buffer_list[2].append(joint_pos)
                    data_list[2] = circular_buffer_list[2].buffer.reshape(num_envs, -1)

                    joint_vel = obs["play_obs"][:, 16:26]
                    circular_buffer_list[3].append(joint_vel)
                    data_list[3] = circular_buffer_list[3].buffer.reshape(num_envs, -1)

                    last_action = obs["play_obs"][:, 26:36]
                    circular_buffer_list[4].append(last_action)
                    data_list[4] = circular_buffer_list[4].buffer.reshape(num_envs, -1)


                    circular_buffer_list[5].append(vel_cmds)
                    data_list[5] = circular_buffer_list[5].buffer.reshape(num_envs, -1)


                    gait_indices = torch.remainder(policy_counter * env.unwrapped.step_dt * gait_cmds[:, 0], 1.0)
                    gait_indices = gait_indices.unsqueeze(-1)
                    sin_phase = torch.sin(2 * torch.pi * gait_indices)
                    cos_phase = torch.cos(2 * torch.pi * gait_indices)
                    gait_phase = torch.cat([sin_phase, cos_phase], dim=-1)
                    circular_buffer_list[6].append(gait_phase)
                    data_list[6] = circular_buffer_list[6].buffer.reshape(num_envs, -1)

                    circular_buffer_list[7].append(gait_cmds)
                    data_list[7] = circular_buffer_list[7].buffer.reshape(num_envs, -1)

                    new_obs = torch.cat(data_list, dim=-1)
                    new_obs_dict = obs.clone()
                    new_obs_dict["policy"] = new_obs
                    # new_obs["policy"][:, 36:39] = vel_cmds
                    # new_obs["policy"][:, 39:41] = gait_phase
                    # new_obs["policy"][:, 41:] = gait_cmds
                    # new_obs["policy"][:, 36:39] = vel_cmds
                    # new_obs["policy"][:, 39:] = gait_cmds
                    actions = policy(new_obs_dict)

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            # actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        counter += 1

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
