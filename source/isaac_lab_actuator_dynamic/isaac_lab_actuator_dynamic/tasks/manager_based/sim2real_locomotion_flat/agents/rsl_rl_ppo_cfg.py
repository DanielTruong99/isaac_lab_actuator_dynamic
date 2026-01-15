from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlPpoActorCriticRecurrentCfg
from isaaclab_rl.rsl_rl import RslRlDistillationAlgorithmCfg, RslRlDistillationStudentTeacherRecurrentCfg, RslRlDistillationStudentTeacherCfg

@configclass
class WalkingRobotPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 100
    experiment_name = "locomotion_robot_flat"   
    empirical_normalization = False

    # resume = True
    # load_checkpoint = "model_50000.pt"
    # load_run = "2025-10-30_14-41-19"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std = 1.0,
        activation = "elu",
        actor_hidden_dims = [512, 256, 128],
        critic_hidden_dims = [512, 256, 128],
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


@configclass
class WalkingRobot_DistillationRunnerCfg(WalkingRobotPPORunnerCfg):
    class_name = "DistillationRunner"
    seed = 42
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    run_name = "distillation"
    
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        gradient_length=5,
        learning_rate=3e-4,
        loss_type="mse",
    )
    # policy = RslRlDistillationStudentTeacherRecurrentCfg(
    #     student_hidden_dims=[256, 256, 128],
    #     teacher_hidden_dims=[256, 256, 128],
    #     activation="elu",
    #     init_noise_std=0.1,
    #     class_name="StudentTeacherRecurrent",
    #     rnn_type="lstm",
    #     rnn_hidden_dim=256,
    #     rnn_num_layers=3,
    #     teacher_recurrent=False,
    # )

    policy = RslRlDistillationStudentTeacherCfg(
        student_hidden_dims=[512, 256, 256],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=0.21,
    )

    def __post_init__(self):
        super().__post_init__()
        # self.max_iterations = 1500


@configclass
class WalkingRobot_StudentPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 100
    experiment_name = "locomotion_robot_flat"   
    empirical_normalization = False

    # resume = True
    # load_checkpoint = "model_50000.pt"
    # load_run = "2025-10-30_14-41-19"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std = 0.1,
        activation = "elu",
        actor_hidden_dims = [512, 256, 256],
        critic_hidden_dims = [512, 256, 128],
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