# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class LeftLegReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 28
    max_iterations = 100000
    save_interval = 50
    experiment_name = "reach_left_leg"
    run_name = ""
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.12,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    # resume = True
    # load_run = "2025-09-05_02-42-31"
    # load_checkpoint = "model_1350.pt"    

@configclass
class LeftLegReachPPORunnerPlayCfg(LeftLegReachPPORunnerCfg):
    resume = True
    load_run = "2025-09-05_02-42-31"
    load_checkpoint = "model_1350.pt"
    # pass
