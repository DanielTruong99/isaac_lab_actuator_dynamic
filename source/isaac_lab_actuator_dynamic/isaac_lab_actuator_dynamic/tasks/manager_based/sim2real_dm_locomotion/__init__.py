import gymnasium as gym

from . import agents

'''
    
'''
gym.register(
    id="Isaac-DMWalkingRobot-Newton-v9",
    entry_point="isaac_lab_actuator_dynamic.tasks.manager_based.sim2real_locomotion.walking_robot:WalkingRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_robot_cfg:DMWalkingRobotNewtonEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkingRobotPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-DMWalkingRobot-Newton-Play-v9",
    entry_point="isaac_lab_actuator_dynamic.tasks.manager_based.sim2real_locomotion.walking_robot:WalkingRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_robot_cfg:DMWalkingRobotNewtonPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkingRobotPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-DMWalkingRobot-Newton-TeacherStudent-v9",
    entry_point="isaac_lab_actuator_dynamic.tasks.manager_based.sim2real_locomotion.walking_robot:WalkingRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_robot_cfg:DMWalkingRobotNewtonEnvCfg_TeacherStudentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkingRobot_VelocityDistillationRunnerCfg",
    },
)

gym.register(
    id="Isaac-DMWalkingRobot-Newton-Student-v9",
    entry_point="isaac_lab_actuator_dynamic.tasks.manager_based.sim2real_locomotion.walking_robot:WalkingRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_robot_cfg:DMWalkingRobotNewtonEnvCfg_StudentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkingRobot_StudentPPORunnerCfg",
    },
)