# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument("--newton_visualizer", action="store_true", default=False, help="Enable Newton rendering.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import time
import numpy as np
from cmaes import CMAwM

import omni
from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from isaac_lab_actuator_dynamic.learning.rsl_rl.runners.custom_on_policy_runner import CustomOnPolicyRunner

from isaaclab.utils.timer import Timer

Timer.enable = False
Timer.enable_display_output = False

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)
import isaac_lab_actuator_dynamic.tasks  # noqa: F401
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.enable_newton_rendering = args_cli.newton_visualizer

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed



    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )


    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env.reset()

    # optimizer configurations
    # theta dimensions: 15
    #    amarture: dim 5
    #    viscous friction: dim 5
    #    coulomb friction: dim 5
    #    joint_pos_offset: dim 0
    mean = np.array([
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        1.5,
        1.5,
        1.5,
        1.5,
        0.8,
        0.43,
        1.0,
        2.0,
        5.7,
        0.8,
        # 0.0,
        # 0.0,
        # 0.0,
        # 0.0,
        # 0.0,
    ])
    bounds = np.array([
        [0.0, 0.5], 
        [0, 0.5], 
        [0, 0.5], 
        [0, 0.5], 
        [0, 0.5], 
        [0.0, 3.0],
        [0.0, 3.0],
        [0.0, 3.0],
        [0.0, 3.0],
        [0.0, 3.0],
        [0.0, 7.0],
        [0.0, 7.0],
        [0.0, 7.0],
        [0.0, 7.0], 
        [0.0, 7.0],
        # [-1.5, 1.5],
        # [-1.5, 1.5],
        # [-1.5, 1.5],
        # [-1.5, 1.5],
        # [-1.5, 1.5],
    ])
    step = np.zeros(15)
    optimizer = CMAwM(
        mean=mean, 
        sigma=1.0, 
        population_size=env_cfg.scene.num_envs, 
        bounds=bounds,
        steps=step
    )

    # runner settings
    # num_steps_per_env = 1500  # TODO fix hardcoding
    step_dt = env.unwrapped.step_dt
    num_steps_per_env = int(0.2 / step_dt)  # 0.2 seconds rollout
    action_dim = env.unwrapped.action_manager._terms["joint_pos"].action_dim
    actions = torch.zeros((env_cfg.scene.num_envs, action_dim), device=env.unwrapped.device)
    obs_dict = env.unwrapped.observation_manager.compute()
    observations = torch.zeros(num_steps_per_env, *obs_dict["policy"].shape, device=env.unwrapped.device)
    rewards = torch.zeros(num_steps_per_env, env.unwrapped.num_envs, 1, device=env.unwrapped.device)

    for epoch in range(100000):
        logs = "="*30 + f" Iteration {epoch} " + "="*30 + "\n"
        logs += f"      Armature | Viscous Friction | Coulomb Friction\n"
        logs += f"mean: {optimizer.mean}\n"

        # sample new parameters
        thetas = []
        eval_thetas = []
        for _ in range(optimizer.population_size):
            x_eval, x_tell = optimizer.ask()
            thetas.append(x_tell)
            eval_thetas.append(x_eval)

        # set the new parameters in the environment
        thetas_np_array = np.stack(thetas)
        thetas = torch.from_numpy(thetas_np_array).to(device=env.unwrapped.device)
        env.unwrapped.set_physic_parameters(thetas) 

        # rollout with the sampled parameters
        with torch.inference_mode():
            start = time.time()

            for step in range(num_steps_per_env):
                obs_dict, rew, terminated, truncated, extras = env.step(actions)
                extras["time_outs"] = truncated

                # move to device
                obs, rew = (obs_dict["policy"].to(env.unwrapped.device), rew.to(env.unwrapped.device))

                # process transition
                observations[step].copy_(obs)
                rewards[step].copy_(rew.clone().view(-1, 1))
            
            stop = time.time()
            collection_time = stop - start
            
            # reset the environment at the time_step
            env.unwrapped.command_manager._terms["motion"].reset_envs(env_ids=None)

            # compute loss
            loss = rewards.mean(dim=0)
            logs += f"avg. loss: {loss.mean()}\n"
            logs += f"Collection time: {collection_time} seconds\n"

        # optimize
        start = time.time()
        
        loss = loss.cpu()
        solution = [pair for pair in zip(thetas.cpu().numpy(), loss)]
        optimizer.tell(solution)

        stop = time.time()
        optimized_time = stop - start
        logs += f"Learning time: {optimized_time} seconds\n"

        print(logs, flush=True)
        if optimizer.should_stop():
            print("Stopping criterion met.")
            print("Best solution found:")
            # print(eval_thetas)
            break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
