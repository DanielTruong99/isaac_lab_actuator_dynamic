
"""Basic IK

Simplest Inverse Kinematics Example using PyRoki.
"""

import time

import numpy as np
import pyroki as pk
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf
from scipy import signal

import pyroki_snippets as pks
import yourdfpy


def calculate_velocities_5point(positions, dt, cutoff_freq=10.0):
    """
    Calculate velocities using 5-point finite difference method with low-pass filter.
    
    The 5-point method uses the formula:
    v[i] = (-f[i+2] + 8*f[i+1] - 8*f[i-1] + f[i-2]) / (12*dt)
    
    Args:
        positions: Array of shape (n_timesteps, n_joints)
        dt: Time step
        cutoff_freq: Low-pass filter cutoff frequency in Hz (default: 10.0)
    
    Returns:
        velocities: Filtered velocities array of same shape as positions
    """
    n_timesteps, n_joints = positions.shape
    velocities = np.zeros_like(positions)
    
    # Calculate velocities for each joint separately
    for joint_idx in range(n_joints):
        pos = positions[:, joint_idx]
        vel = np.zeros_like(pos)
        
        # Use 5-point finite difference for interior points
        for i in range(2, n_timesteps - 2):
            vel[i] = (-pos[i+2] + 8*pos[i+1] - 8*pos[i-1] + pos[i-2]) / (12*dt)
        
        # Handle boundary points using forward/backward differences
        # First two points (forward difference)
        vel[0] = (-3*pos[0] + 4*pos[1] - pos[2]) / (2*dt)
        vel[1] = (pos[2] - pos[0]) / (2*dt)
        
        # Last two points (backward difference)
        vel[-2] = (pos[-1] - pos[-3]) / (2*dt)
        vel[-1] = (3*pos[-1] - 4*pos[-2] + pos[-3]) / (2*dt)
        
        # Apply low-pass filter (Butterworth filter)
        # Use the provided cutoff frequency
        fs = 1.0 / dt  # Sampling frequency
        nyquist_freq = fs / 2.0
        normalized_cutoff = cutoff_freq / nyquist_freq
        
        # Ensure cutoff frequency is within valid range
        if normalized_cutoff >= 1.0:
            print(f"Warning: Cutoff frequency {cutoff_freq} Hz too high for sampling rate {fs} Hz. Using {nyquist_freq * 0.9:.1f} Hz instead.")
            normalized_cutoff = 0.9  # Use 90% of Nyquist frequency
        
        # Design low-pass filter
        b, a = signal.butter(4, normalized_cutoff, btype='low')
        
        # Apply filter (zero-phase filtering)
        vel_filtered = signal.filtfilt(b, a, vel)
        
        velocities[:, joint_idx] = vel_filtered
    
    return velocities


def main():
    """Main function for basic IK."""

    urdf = yourdfpy.URDF.load(
        "isaac_lab_actuator_dynamic/source/isaac_lab_actuator_dynamic/isaac_lab_actuator_dynamic/assets/urdf/leg05/hr.urdf",
        mesh_dir="isaac_lab_actuator_dynamic/source/isaac_lab_actuator_dynamic/isaac_lab_actuator_dynamic/assets/urdf/leg05/meshes",
    )
    target_link_name = "L_toe"

    # Create robot.
    robot = pk.Robot.from_urdf(urdf)

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Create interactive controller with initial position.
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
    
    # Add GUI control for cutoff frequency
    cutoff_freq_handle = server.gui.add_slider(
        "Velocity Filter Cutoff (Hz)", 1.0, 50.0, 0.1, 10.0
    )

    # Create set of ik target
    # a = 0.11; b = 0.1;
    # w = 3.0; n = int(2/0.001);  # 12 seconds at 1000 Hz
    # t = np.linspace(0, 2, n)
    # xo = 0.1; zo = -0.61;
    # x = a * np.cos(w * t) + xo
    # z = b * np.sin(w * t) + zo
    # y = 0.1175
    # pos = np.stack([x, np.full(n, y), z], axis=1)
    # orient = np.zeros((n, 4)); orient[:, 0] = 1.0
    # ee_goals = np.concatenate([pos, orient], axis=1)

    a = 0.11; b = 0.1; c = 0.08;
    t_max = 25;
    n = int(t_max/0.002);
    t = np.linspace(0, t_max, n)
    A_phi = 12; w_phi = 1.7
    A_theta = 3.4; w_theta = 0.265
    xo = 0.11; zo = -0.61; yo = 0.1175
    phi_0 = -A_phi / w_phi; theta_0 = -A_phi / w_phi
    phi = -A_phi / w_phi * np.cos(w_phi * t) + A_phi / w_phi + phi_0
    theta = -A_theta / w_theta * np.cos(w_theta * t) + A_theta / w_theta + theta_0
    x = a * np.cos(theta) * np.cos(phi) + xo
    y = b * np.cos(theta) * np.sin(phi) + yo
    z = c * np.sin(theta) + zo
    pos = np.stack([x, y, z], axis=1)
    orient = np.zeros((n, 4)); orient[:, 0] = 1.0
    ee_goals = np.concatenate([pos, orient], axis=1)

    # Add batched axes to visualize the trajectory
    trajectory_viz_handle = server.scene.add_batched_axes(
        "trajectory_visualization",
        axes_length=0.02,
        axes_radius=0.002,
        batched_positions=pos,  # Use the positions from ee_goals
        batched_wxyzs=orient,   # Use the orientations from ee_goals
    )


    # Storage for timing results and trajectory index
    timing_results = []
    trajectory_index = 0
    
    # Storage for motion data
    dof_positions_list = []
    all_dof_names = robot.joints.actuated_names  # Get all actuated joint names
    # Filter out R_... joints (exclude right-side joints)
    dof_names = [name for name in all_dof_names if not name.startswith('R_')]
    # Get indices of the filtered joints
    dof_indices = [all_dof_names.index(name) for name in dof_names]
    
    print(f"Excluding R_... joints. Using {len(dof_names)} out of {len(all_dof_names)} joints:")
    print(f"Selected joints: {dof_names}")
    
    while True:
        # Get current trajectory point
        ee_goal = ee_goals[trajectory_index]
        target_position = ee_goal[:3]  # First 3 elements are position
        target_wxyz = ee_goal[3:]      # Last 4 elements are orientation (w,x,y,z)
        
        # Solve IK and time it
        start_time = time.time()
        solution = pks.solve_ik(
            robot=robot,
            target_link_name=target_link_name,
            target_position=target_position,
            target_wxyz=target_wxyz,
        )
        elapsed_time = time.time() - start_time
        
        # Store timing result and joint positions
        timing_results.append(elapsed_time * 1000)  # Convert to milliseconds
        # Store only the filtered joint configurations (exclude R_... joints)
        filtered_solution = solution[dof_indices]
        dof_positions_list.append(filtered_solution.copy())
        
        # Update timing handle
        timing_handle.value = elapsed_time * 1000
        
        # Update visualizer
        urdf_vis.update_cfg(solution)
        
        # Print timing info for current point
        # print(f"Point {trajectory_index+1}/{n}: Position={target_position}, Time={elapsed_time*1000:.3f}ms")
        
        # Move to next trajectory point
        trajectory_index = (trajectory_index + 1) % n
        
        # Check if we completed one full trajectory
        if trajectory_index == 0 and len(dof_positions_list) >= n:
            print(f"Completed trajectory with {len(dof_positions_list)} points")
            break
        
        # Add real-time delay (1ms)
        time.sleep(0.001)
    
    # Convert to numpy arrays
    dof_positions = np.array(dof_positions_list)
    
    # Calculate velocities using 5-point finite difference method with low-pass filter
    dt = 0.002  # 2ms timestep from sleep
    cutoff_freq = 100  # Get cutoff frequency from GUI slider
    print(f"Calculating velocities using 5-point method and applying low-pass filter (cutoff: {cutoff_freq} Hz)...")
    dof_velocities = calculate_velocities_5point(dof_positions, dt, cutoff_freq)
    
    # Calculate FPS
    fps = 1.0 / dt  # 500 Hz

    # Create data dictionary
    data_dict = {
        "fps": fps,
        "dof_names": dof_names,
        "dof_positions": dof_positions,
        "dof_velocities": dof_velocities,
    }
    
    # Save to NPZ file
    filename = "ik_trajectory_data_2.npz"
    np.savez(filename, **data_dict)
    print(f"Saved trajectory data to {filename}")
    print(f"Data shape: positions={dof_positions.shape}, velocities={dof_velocities.shape}")
    print(f"DOF names: {dof_names}")
    print(f"FPS: {fps}")
    
    # Print summary statistics
    timing_array = np.array(timing_results)
    print(f"\nTiming Summary:")
    print(f"Mean: {np.mean(timing_array):.3f}ms")
    print(f"Std:  {np.std(timing_array):.3f}ms")
    print(f"Min:  {np.min(timing_array):.3f}ms")
    print(f"Max:  {np.max(timing_array):.3f}ms")


if __name__ == "__main__":
    main()
