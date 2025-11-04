import gymnasium as gym

from . import agents

'''
    
'''
gym.register(
    id="Isaac-RoughWalkingRobot-Newton-v9",
    entry_point="isaac_lab_actuator_dynamic.tasks.manager_based.sim2real_locomotion.walking_robot:WalkingRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_robot_cfg:WalkingRobotNewtonEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkingRobotPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-RoughWalkingRobot-Newton-Play-v9",
    entry_point="isaac_lab_actuator_dynamic.tasks.manager_based.sim2real_locomotion.walking_robot:WalkingRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_robot_cfg:WalkingRobotNewtonPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkingRobotPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-RoughWalkingRobot-Newton-FineTune-v9",
    entry_point="isaac_lab_actuator_dynamic.tasks.manager_based.sim2real_locomotion.walking_robot:WalkingRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_robot_cfg:WalkingRobotNewtonEnvFineTuneCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkingRobotPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-RoughWalkingRobot-Newton-FineTune-Play-v9",
    entry_point="isaac_lab_actuator_dynamic.tasks.manager_based.sim2real_locomotion.walking_robot:WalkingRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_robot_cfg:WalkingRobotNewtonEnvFineTuneCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkingRobotPPORunnerCfg",
    },
)