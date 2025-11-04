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
    id="Isaac-RoughWalkingRobot-v5",
    entry_point="isaac_lab_actuator_dynamic.tasks.manager_based.locomotion.walking_robot:WalkingRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_robot_cfg:WalkingRobotEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkingRobotPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-RoughWalkingRobot-Newton-v5",
    entry_point="isaac_lab_actuator_dynamic.tasks.manager_based.locomotion.walking_robot:WalkingRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_robot_cfg:WalkingRobotNewtonEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkingRobotPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-RoughWalkingRobot-Newton-Play-v5",
    entry_point="isaac_lab_actuator_dynamic.tasks.manager_based.locomotion.walking_robot:WalkingRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_robot_cfg:WalkingRobotNewtonPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkingRobotPPORunnerPlayCfg",
    },
)

gym.register(
    id="Isaac-RoughWalkingRobot-Distillation-Newton-v5",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_robot_finetune_cfg:WalkingRobot_FlatTeacherStudentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkingRobot_VelocityDistillationRunnerCfg",
    },
)

gym.register(
    id="Isaac-RoughWalkingRobot-Student-Finetune-Newton-v5",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_robot_finetune_cfg:WalkingRobot_FlatStudentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkingRobot_VelocityStudentFinetuneRunnerCfg",
    },
)

gym.register(
    id="Isaac-RoughWalkingRobot-Student-Finetune-Newton-Play-v5",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_robot_finetune_cfg:WalkingRobot_FlatStudentPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkingRobot_VelocityStudentFinetuneRunnerCfg",
    },
)

gym.register(
    id="Isaac-RoughWalkingRobot-FineTune-v5",
    entry_point="isaac_lab_actuator_dynamic.tasks.manager_based.locomotion.walking_robot:WalkingRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_robot_cfg:WalkingRobotEnvFinetuneCfg",
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

gym.register(
    id="Isaac-RoughWalkingRobot-FineTune-Play-v5",
    entry_point="isaac_lab_actuator_dynamic.tasks.manager_based.locomotion.walking_robot:WalkingRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_robot_cfg:WalkingRobotPLayFinetuneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkingRobotPPORunnerCfg",
    },
)
