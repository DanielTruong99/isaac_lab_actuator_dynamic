from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlPpoActorCriticRecurrentCfg

@configclass
class WalkingRobotPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 50
    experiment_name = "walking_robot_unitree"   
    empirical_normalization = False

    # resume = True
    # load_checkpoint = "model_10000.pt"
    # load_run = "2025-09-20_17-11-53"

    # policy = RslRlPpoActorCriticRecurrentCfg(
    #     init_noise_std = 1.0,
    #     activation = "elu",
    #     actor_hidden_dims = [256, 256, 256],
    #     critic_hidden_dims = [256, 256, 256],
    #     rnn_type = "gru",
    #     rnn_hidden_dim = 64,
    #     rnn_num_layers = 1
    # )

    policy = RslRlPpoActorCriticCfg(
        init_noise_std = 1.0,
        activation = "elu",
        actor_hidden_dims = [256, 256, 256],
        critic_hidden_dims = [256, 256, 256],
        # rnn_type = "gru",
        # rnn_hidden_dim = 64,
        # rnn_num_layers = 1
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )