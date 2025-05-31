# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
AMP Humanoid locomotion environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Actuator-Dynamic-Direct-v0",
    entry_point=f"{__name__}.actuator_dynamic_amp_env:ActuatorDynamicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.actuator_dynamic_amp_env_cfg:ActuatorDynamicEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_actuator_dynamic_amp_cfg.yaml",
    },
)