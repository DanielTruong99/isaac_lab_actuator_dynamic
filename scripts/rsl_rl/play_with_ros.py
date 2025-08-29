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
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
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
import numpy as np

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import isaac_lab_actuator_dynamic.tasks  # noqa: F401
# PLACEHOLDER: Extension template (do not remove this comment)

# from isaacsim.core.utils.extensions import enable_extension
# enable_extension("isaacsim.ros2.bridge")
# simulation_app.update()


# import rclpy
# from sensor_msgs.msg import JointState

# rclpy.init()
# node = rclpy.create_node('Humanoid_HWITL')
# pub_joint_states = node.create_publisher(JointState, '/joint_states', 10)
# pub_joint_cmds = node.create_publisher(JointState, '/joint_cmds', 10)
# joint_states_msg = JointState()
# joint_cmds_msg = JointState()

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

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

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

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
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt
    
    # reset environment
    obs, _ = env.get_observations()
    joint_data = []
    joint_cmd_data = []
    joint_key_ids = env.unwrapped.key_joint_indexes
    joint_position = env.unwrapped.robot.data.joint_pos[:, joint_key_ids].cpu().numpy().tolist()
    joint_data += joint_position
    joint_cmd = env.unwrapped.joint_pos_cmds.cpu().numpy().tolist()
    joint_cmd_data += joint_cmd
    timestep = 0

    residual_torque_data = []
    applied_torque_data = []
    
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)

        # # get joint states
        joint_key_ids = env.unwrapped.key_joint_indexes
        joint_position = env.unwrapped.robot.data.joint_pos[:, joint_key_ids].cpu().numpy().tolist()
        joint_data += joint_position

        # get joint commands
        joint_pos_cmd = env.unwrapped.joint_pos_cmds.cpu().numpy().tolist()
        joint_cmd_data += joint_pos_cmd

        # get residual torques
        residual_torques = env.unwrapped.residual_torques.cpu().numpy().tolist()
        residual_torque_data += residual_torques

        # get applied torques
        applied_torques = env.unwrapped.robot.data.applied_torque[:, joint_key_ids].cpu().numpy().tolist()
        applied_torque_data += applied_torques

        if dones.any():
            print("[INFO] Episode done, resetting the environment...")
            # plot the  joint positions
            joint_data = np.array(joint_data)
            joint_cmd_data = np.array(joint_cmd_data)
            residual_torque_data = np.array(residual_torque_data)
            applied_torque_data = np.array(applied_torque_data)
            time_data = np.arange(joint_data.shape[0]) * dt
            for i in range(joint_data.shape[1]):
                plt.figure()
                plt.plot(time_data, joint_data[:, i], label='Joint Position')
                plt.plot(time_data, joint_cmd_data[:, i], label='Joint Command', linestyle='--')
                plt.xlabel('Time (s)')
                plt.ylabel('Position (rad)')
                plt.title(f'Joint {i} Position vs Command')
                plt.legend()
                plt.grid()

            time_data = np.arange(residual_torque_data.shape[0]) * dt
            for i in range(residual_torque_data.shape[1]):
                plt.figure()
                plt.plot(time_data, residual_torque_data[:, i], label='Residual Torque')
                plt.plot(time_data, applied_torque_data[:, i], label='Applied Torque', linestyle='--')
                plt.xlabel('Time (s)')
                plt.ylabel('Torque (Nm)')
                plt.title(f'Joint {i} Residual vs Applied Torque')
                plt.legend()
                plt.grid()
            plt.show()

            joint_data = []
            joint_cmd_data = []

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

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
