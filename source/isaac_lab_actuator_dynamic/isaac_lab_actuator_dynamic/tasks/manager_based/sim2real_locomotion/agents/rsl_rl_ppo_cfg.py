from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlPpoActorCriticRecurrentCfg
from isaaclab_rl.rsl_rl import RslRlDistillationAlgorithmCfg, RslRlDistillationStudentTeacherRecurrentCfg, RslRlDistillationStudentTeacherCfg

@configclass
class WalkingRobotPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 100
    experiment_name = "locomotion_robot"   
    empirical_normalization = False

    # resume = True
    # load_checkpoint = "model_50000.pt"
    # load_run = "2025-10-30_14-41-19"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std = 1.0,
        activation = "elu",
        actor_hidden_dims = [256, 256, 256],
        critic_hidden_dims = [256, 256, 256],
    )
    # policy = RslRlPpoActorCriticCfg(
    #     init_noise_std = 1.0,
    #     activation = "elu",
    #     actor_hidden_dims = [256, 256, 256],
    #     critic_hidden_dims = [256, 256, 256],
    # )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class WalkingRobotPPORunnerPlayCfg(WalkingRobotPPORunnerCfg):
    # resume = True
    # load_checkpoint = "model_107000.pt"
    # load_run = "2025-10-12_04-04-23"
    pass
