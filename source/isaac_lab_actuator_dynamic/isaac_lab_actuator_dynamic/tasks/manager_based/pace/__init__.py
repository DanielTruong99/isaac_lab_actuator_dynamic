import gymnasium as gym

from . import agents



gym.register(
    id="Isaac-Newton-Pace-v0",
    entry_point="isaac_lab_actuator_dynamic.tasks.manager_based.pace.pace_env:PaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pace_env_cfg:PaceNewtonEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkingRobotPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Newton-Eval-Pace-v0",
    entry_point="isaac_lab_actuator_dynamic.tasks.manager_based.pace.pace_env:PaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pace_env_cfg:PaceNewtonEvalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WalkingRobotPPORunnerCfg",
    },
)

