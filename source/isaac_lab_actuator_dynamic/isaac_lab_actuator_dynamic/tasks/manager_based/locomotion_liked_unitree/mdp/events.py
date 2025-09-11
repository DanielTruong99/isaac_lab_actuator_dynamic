import torch

##
#! User defined configs
##
from ..walking_robot import WalkingRobotEnv

def update_phase(env: WalkingRobotEnv, env_ids: torch.Tensor):
    """
    Update the phase of the walking robot environment.
    This function calculates and updates the phase of the walking robot's gait cycle.
    The phase is determined based on the episode length buffer and the step duration.
    It also calculates the phase for the left and right legs of the robot.
    Args:
        env (WalkingRobotEnv): The walking robot environment instance.
    Attributes:
        env.phase (float): The current phase of the robot's gait cycle.
        env.phase_left (float): The phase of the left leg.
        env.phase_right (float): The phase of the right leg.
    """

    period = 0.8
    offset = 0.5
    env.phase[env_ids] = (env.episode_length_buf[env_ids] * env.step_dt) % period / period
    env.phase_left[env_ids] = env.phase[env_ids]
    env.phase_right[env_ids] = (env.phase[env_ids] + offset) % 1