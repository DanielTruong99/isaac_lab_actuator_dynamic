import torch

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv

##
#! User defined configs
##
# from .walking_robot_cfg import WalkingRobotEnvCfg


class WalkingRobotEnv(ManagerBasedRLEnv):
    
    def __init__(self, cfg, render_mode: str | None = None, **kwargs): # type: ignore

        # Init phase buffer
        self.phase = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device, dtype=torch.float32, requires_grad=False)
        self.phase_left = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device, dtype=torch.float32, requires_grad=False)
        self.phase_right = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device, dtype=torch.float32, requires_grad=False)

        # Init 
        super().__init__(cfg=cfg, render_mode=render_mode)

