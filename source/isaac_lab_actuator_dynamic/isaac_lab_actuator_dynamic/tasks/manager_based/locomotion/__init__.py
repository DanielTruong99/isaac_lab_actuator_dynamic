import gymnasium as gym

from . import agents

'''
    Register the RoughWalkingRobot-v5  and RoughWalkingRobot-Play-v5
'''
gym.register(
    id="Isaac-RoughWalkingRobot-v5",
    entry_point="isaac_lab_actuator_dynamic.tasks.manager_based.locomotion.walking_robot:WalkingRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_robot_cfg:WalkingRobotEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkingRobotPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-RoughWalkingRobot-Play-v5",
    entry_point="isaac_lab_actuator_dynamic.tasks.manager_based.locomotion.walking_robot:WalkingRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_robot_cfg:WalkingRobotPLayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkingRobotPPORunnerCfg",
    },
)
