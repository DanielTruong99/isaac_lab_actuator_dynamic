import gymnasium as gym

from . import agents

'''
    
'''
gym.register(
    id="Isaac-WalkingRobot-Newton-v9",
    entry_point="isaac_lab_actuator_dynamic.tasks.manager_based.sim2real_locomotion_flat.walking_robot:WalkingRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_robot_cfg:WalkingRobotNewtonEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkingRobotPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-WalkingRobotRough-Newton-v9",
    entry_point="isaac_lab_actuator_dynamic.tasks.manager_based.sim2real_locomotion_flat.walking_robot:WalkingRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_robot_cfg:RoughWalkingRobotNewtonEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkingRobotPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-WalkingRobot-Newton-Play-v9",
    entry_point="isaac_lab_actuator_dynamic.tasks.manager_based.sim2real_locomotion_flat.walking_robot:WalkingRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_robot_cfg:WalkingRobotNewtonPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkingRobotPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-WalkingRobot-TeacherStudent-Newton-v9",
    entry_point="isaac_lab_actuator_dynamic.tasks.manager_based.sim2real_locomotion_flat.walking_robot:WalkingRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_robot_finetune_cfg:WalkingRobot_TeacherStudentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkingRobot_DistillationRunnerCfg",
    },
)

gym.register(
    id="Isaac-WalkingRobot-TeacherStudent-Newton-Play-v9",
    entry_point="isaac_lab_actuator_dynamic.tasks.manager_based.sim2real_locomotion_flat.walking_robot:WalkingRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_robot_finetune_cfg:WalkingRobot_TeacherStudentPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkingRobot_DistillationRunnerCfg",
    },
)

gym.register(
    id="Isaac-WalkingRobot-Student-Newton-v9",
    entry_point="isaac_lab_actuator_dynamic.tasks.manager_based.sim2real_locomotion_flat.walking_robot:WalkingRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_robot_finetune_cfg:WalkingRobot_StudentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkingRobot_StudentPPORunnerCfg",
    },
)

