import torch

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.assets import Articulation



class PaceEnv(ManagerBasedRLEnv):
    
    def set_physic_parameters(self, thetas):
        """
            thetas: num_envs x 15, 
                amarture: dim 5
                viscous: dim 5
                coulomb: dim 5
        """
        armature, viscous, coulomb = torch.split(thetas, [5, 5, 5], dim=1)

        self.action_manager._terms["joint_pos"].set_joint_frictions(coulomb=coulomb, viscous=viscous)
        
        joint_ids = self.command_manager._terms["motion"]._joint_ids
        armature = armature.to(dtype=torch.float32)
        self.scene["robot"].write_joint_armature_to_sim(armature=armature, joint_ids=joint_ids)

