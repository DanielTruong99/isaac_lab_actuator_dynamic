import gymnasium as gym

from . import agents

'''
    
'''
gym.register(
    id="Isaac-WalkingRobot-v9",
    entry_point="isaac_lab_actuator_dynamic.tasks.manager_based.sim2real_locomotion_flat.walking_robot:WalkingRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_robot_cfg:WalkingRobotEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkingRobotPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-WalkingRobot-Play-v9",
    entry_point="isaac_lab_actuator_dynamic.tasks.manager_based.sim2real_locomotion_flat.walking_robot:WalkingRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_robot_cfg:WalkingRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkingRobotPPORunnerCfg",
    },
)


