"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python replay_motion.py --motion_file source/whole_body_tracking/whole_body_tracking/assets/g1/motions/lafan_walk_short.npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
# parser.add_argument("--registry_name", type=str, required=True, help="The name of the wand registry.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from isaac_lab_actuator_dynamic.tasks.manager_based.sim2real_dm_locomotion.dm_mdp.commands_3 import MotionLoader
from isaac_lab_actuator_dynamic.assets.leg_walking import LEGWALKING_HIGH_GAIN_AMARTURE_5_CFG

@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = LEGWALKING_HIGH_GAIN_AMARTURE_5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    # sim_dt = sim.get_physics_dt()
    sim_dt = 1/40.0

    motion_folder = "source/isaac_lab_actuator_dynamic/isaac_lab_actuator_dynamic/tasks/manager_based/sim2real_dm_locomotion/motions/"
    motion_name = "07_02_poses.npz"
    motion_file = [motion_folder + motion_name]

    motion = MotionLoader(
        motion_file,
        1,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    dof_positions = []
    dof_vels = []
    body_positions_w = []
    body_quats_w = []
    body_lin_vels_w = []
    body_ang_vels_w = []
    is_first = True

    start_id = 0
    end_id = motion.time_step_total[0]
    # end_id = 170

    time_steps[:] = start_id
    motion.time_step_total[0] = end_id

    # Simulation loop
    while simulation_app.is_running():
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total[0]
        time_steps[reset_ids] = start_id
        if is_first and torch.any(reset_ids):
            stacked_dof_positions = np.concatenate(dof_positions, axis=0)
            stacked_dof_vels =  np.concatenate(dof_vels, axis=0)
            stacked_body_positions_w =  np.concatenate(body_positions_w, axis=0)
            stacked_body_quats_w =  np.concatenate(body_quats_w, axis=0)
            stacked_body_lin_vels_w =  np.concatenate(body_lin_vels_w, axis=0)
            stacked_body_ang_vels_w =  np.concatenate(body_ang_vels_w, axis=0)

            print("max linear vel:", np.max(stacked_body_lin_vels_w[:, 0, :]))
            print("max angular vel:", np.max(stacked_body_ang_vels_w[:, 0, :]))
            print("max dof vel:", np.max(stacked_dof_vels))
            print("num frames:", stacked_dof_positions.shape[0])

            np.savez(
                "walk_straight_4.npz",
                dof_positions=stacked_dof_positions,
                dof_vels=stacked_dof_vels,
                body_positions_w=stacked_body_positions_w,
                body_quats_w=stacked_body_quats_w,
                body_lin_vels_w=stacked_body_lin_vels_w,
                body_ang_vels_w=stacked_body_ang_vels_w,
                fps=40,       
            )
            print("Saved replayed motion data to 'replayed_motion_data.npz'")
            is_first = False

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins[:, None, :]
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        # Joint position adjustment
        # Lhip Lhip2 L_thigh Lcalf Ltoe
        #  0     1      2     3     4
        # Lhip Lhip2 L_thigh Lcalf Ltoe
        #  5     6      7     8     9
        dof_pos_ref = motion.joint_pos[time_steps]; dof_pos_robot = dof_pos_ref.clone() 
        dof_pos_robot[:, 1] = -dof_pos_ref[:, 1]
        dof_pos_robot[:, 6] = -dof_pos_ref[:, 6]

        dof_pos_robot[:, 2] = np.pi/4 + dof_pos_ref[:, 2]
        dof_pos_robot[:, 7] = np.pi/4 + dof_pos_ref[:, 7]

        dof_pos_robot[:, 3] = -dof_pos_ref[:, 3] - np.pi/2
        dof_pos_robot[:, 8] = -dof_pos_ref[:, 8] - np.pi/2

        dof_pos_robot[:, 4] = dof_pos_ref[:, 4] + np.pi/4
        dof_pos_robot[:, 9] = dof_pos_ref[:, 9] + np.pi/4

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(dof_pos_robot, motion.joint_vel[time_steps])
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)

        dof_positions.append(robot.data.joint_pos.cpu().numpy())
        dof_vels.append(robot.data.joint_vel.cpu().numpy())
        body_positions_w.append(robot.data.body_pos_w.cpu().numpy())
        body_quats_w.append(robot.data.body_quat_w.cpu().numpy())
        body_lin_vels_w.append(robot.data.body_lin_vel_w.cpu().numpy())
        body_ang_vels_w.append(robot.data.body_ang_vel_w.cpu().numpy())

        # pos_lookat = root_states[0, :3].cpu().numpy()
        # sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)
        print(f"Time step: {time_steps[0].item()}")


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1/40.0  # Set sim to 40Hz
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()