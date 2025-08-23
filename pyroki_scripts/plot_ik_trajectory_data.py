"""Plot IK Trajectory Data

Script to read and visualize the exported NPZ file from IK trajectory generation.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
from scipy import signal
import pandas as pd


def plot_trajectory_data(filename="ik_trajectory_data.npz"):
    """
    Load and plot trajectory data from NPZ file.
    
    Args:
        filename: Path to the NPZ file containing trajectory data
    """
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found!")
        return
    
    # Load data
    print(f"Loading data from {filename}...")
    data = np.load(filename)
    
    # Extract data
    fps = data["fps"]
    dof_names = data["dof_names"]
    dof_positions = data["dof_positions"]
    dof_velocities = data["dof_velocities"]
    
    # Print data info
    print(f"FPS: {fps}")
    print(f"Number of DOFs: {len(dof_names)}")
    print(f"DOF names: {list(dof_names)}")
    print(f"Data shape: positions={dof_positions.shape}, velocities={dof_velocities.shape}")
    print(f"Trajectory duration: {dof_positions.shape[0] / fps:.3f} seconds")
    
    # Create time array
    num_timesteps = dof_positions.shape[0]
    time = np.arange(num_timesteps) / fps
    
    # Create subplots
    num_dofs = len(dof_names)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot positions
    ax1 = axes[0]
    for i, dof_name in enumerate(dof_names):
        ax1.plot(time, dof_positions[:, i], label=dof_name, linewidth=1.5)
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Joint Position (rad)')
    ax1.set_title('Joint Positions Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot velocities
    ax2 = axes[1]
    for i, dof_name in enumerate(dof_names):
        ax2.plot(time, dof_velocities[:, i], label=dof_name, linewidth=1.5)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Joint Velocity (rad/s)')
    ax2.set_title('Joint Velocities Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Additional analysis plots
    plot_joint_analysis(dof_names, dof_positions, dof_velocities, time)


def plot_joint_analysis(dof_names, dof_positions, dof_velocities, time):
    """
    Create additional analysis plots for each joint.
    """
    
    num_dofs = len(dof_names)
    
    # Create individual joint plots
    fig, axes = plt.subplots(num_dofs, 2, figsize=(15, 4 * num_dofs))
    if num_dofs == 1:
        axes = axes.reshape(1, -1)
    
    for i, dof_name in enumerate(dof_names):
        # Position plot
        ax_pos = axes[i, 0]
        ax_pos.plot(time, dof_positions[:, i], 'b-', linewidth=2)
        ax_pos.set_xlabel('Time (s)')
        ax_pos.set_ylabel('Position (rad)')
        ax_pos.set_title(f'{dof_name} - Position')
        ax_pos.grid(True, alpha=0.3)
        
        # Add statistics
        pos_mean = np.mean(dof_positions[:, i])
        pos_std = np.std(dof_positions[:, i])
        pos_range = np.ptp(dof_positions[:, i])
        ax_pos.axhline(y=pos_mean, color='r', linestyle='--', alpha=0.7, label=f'Mean: {pos_mean:.3f}')
        ax_pos.text(0.02, 0.95, f'Range: {pos_range:.3f} rad\nStd: {pos_std:.3f} rad', 
                   transform=ax_pos.transAxes, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Velocity plot
        ax_vel = axes[i, 1]
        ax_vel.plot(time, dof_velocities[:, i], 'g-', linewidth=2)
        ax_vel.set_xlabel('Time (s)')
        ax_vel.set_ylabel('Velocity (rad/s)')
        ax_vel.set_title(f'{dof_name} - Velocity')
        ax_vel.grid(True, alpha=0.3)
        
        # Add statistics
        vel_mean = np.mean(dof_velocities[:, i])
        vel_std = np.std(dof_velocities[:, i])
        vel_max = np.max(np.abs(dof_velocities[:, i]))
        ax_vel.axhline(y=vel_mean, color='r', linestyle='--', alpha=0.7, label=f'Mean: {vel_mean:.3f}')
        ax_vel.text(0.02, 0.95, f'Max |vel|: {vel_max:.3f} rad/s\nStd: {vel_std:.3f} rad/s', 
                   transform=ax_vel.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


def plot_3d_histogram(filename="ik_trajectory_data.npz", bins=30):
    """
    Create 3D histograms showing joint position vs velocity distributions.
    
    Args:
        filename: Path to the NPZ file containing trajectory data
        bins: Number of bins for histograms
    """
    
    data = np.load(filename)
    dof_names = data["dof_names"]
    dof_positions = data["dof_positions"]
    dof_velocities = data["dof_velocities"]
    
    num_dofs = len(dof_names)
    
    # Create 3D subplots
    cols = min(2, num_dofs)
    rows = (num_dofs + cols - 1) // cols
    
    fig = plt.figure(figsize=(12 * cols, 8 * rows))
    
    for i, dof_name in enumerate(dof_names):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        
        pos_data = dof_positions[:, i]
        vel_data = dof_velocities[:, i]
        
        # Create 2D histogram
        hist, xedges, yedges = np.histogram2d(pos_data, vel_data, bins=bins)
        
        # Create meshgrid for 3D surface
        xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')
        
        # Flatten for 3D bar plot
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = np.zeros_like(xpos)
        
        dx = (xedges[1] - xedges[0]) * np.ones_like(zpos)
        dy = (yedges[1] - yedges[0]) * np.ones_like(zpos)
        dz = hist.ravel()
        
        # Create 3D bar plot
        colors = plt.cm.viridis(dz / np.max(dz))
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=0.8)
        
        ax.set_xlabel('Joint Position (rad)')
        ax.set_ylabel('Joint Velocity (rad/s)')
        ax.set_zlabel('Frequency')
        ax.set_title(f'{dof_name} - 3D Position-Velocity Distribution')
        
        # Add statistics text
        pos_mean = np.mean(pos_data)
        vel_mean = np.mean(vel_data)
        ax.text2D(0.05, 0.95, f'Pos Mean: {pos_mean:.3f}\nVel Mean: {vel_mean:.3f}', 
                 transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


def plot_3d_surface_histogram(filename="ik_trajectory_data.npz", bins=30):
    """
    Create 3D surface histograms showing joint position vs velocity distributions.
    
    Args:
        filename: Path to the NPZ file containing trajectory data
        bins: Number of bins for histograms
    """
    
    data = np.load(filename)
    dof_names = data["dof_names"]
    dof_positions = data["dof_positions"]
    dof_velocities = data["dof_velocities"]
    
    num_dofs = len(dof_names)
    
    # Create 3D subplots
    cols = min(2, num_dofs)
    rows = (num_dofs + cols - 1) // cols
    
    fig = plt.figure(figsize=(12 * cols, 8 * rows))
    
    for i, dof_name in enumerate(dof_names):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        
        pos_data = dof_positions[:, i]
        vel_data = dof_velocities[:, i]
        
        # Create 2D histogram
        hist, xedges, yedges = np.histogram2d(pos_data, vel_data, bins=bins)
        
        # Create meshgrid for 3D surface
        X, Y = np.meshgrid(xedges[:-1] + (xedges[1] - xedges[0])/2, 
                          yedges[:-1] + (yedges[1] - yedges[0])/2)
        
        # Create 3D surface plot
        surf = ax.plot_surface(X, Y, hist.T, cmap='viridis', alpha=0.8, 
                              linewidth=0, antialiased=True)
        
        # Add contour lines at the bottom
        ax.contour(X, Y, hist.T, zdir='z', offset=0, cmap='viridis', alpha=0.5)
        
        ax.set_xlabel('Joint Position (rad)')
        ax.set_ylabel('Joint Velocity (rad/s)')
        ax.set_zlabel('Frequency')
        ax.set_title(f'{dof_name} - 3D Surface Distribution')
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        
        # Add statistics text
        pos_mean = np.mean(pos_data)
        vel_mean = np.mean(vel_data)
        max_freq = np.max(hist)
        ax.text2D(0.05, 0.95, f'Pos Mean: {pos_mean:.3f}\nVel Mean: {vel_mean:.3f}\nMax Freq: {max_freq:.0f}', 
                 transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


def plot_spectrum_analysis(filename="ik_trajectory_data.npz"):
    """
    Load and plot trajectory data from NPZ file.
    
    Args:
        filename: Path to the NPZ file containing trajectory data
    """
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found!")
        return
    
    # Load data
    print(f"Loading data from {filename}...")
    data = np.load(filename)
    
    # Extract data
    fps = data["fps"]
    dof_names = data["dof_names"]
    dof_positions = data["dof_positions"]
    dof_velocities = data["dof_velocities"]
    
    # Print data info
    print(f"FPS: {fps}")
    print(f"Number of DOFs: {len(dof_names)}")
    print(f"DOF names: {list(dof_names)}")
    print(f"Data shape: positions={dof_positions.shape}, velocities={dof_velocities.shape}")
    print(f"Trajectory duration: {dof_positions.shape[0] / fps:.3f} seconds")
    
    # Create time array
    num_timesteps = dof_positions.shape[0]
    time = np.arange(num_timesteps) / fps
    
    # Create subplots
    num_dofs = len(dof_names)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot positions
    ax1 = axes[0]
    for i, dof_name in enumerate(dof_names):
        ax1.plot(time, dof_positions[:, i], label=dof_name, linewidth=1.5)
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Joint Position (rad)')
    ax1.set_title('Joint Positions Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot velocities
    ax2 = axes[1]
    for i, dof_name in enumerate(dof_names):
        ax2.plot(time, dof_velocities[:, i], label=dof_name, linewidth=1.5)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Joint Velocity (rad/s)')
    ax2.set_title('Joint Velocities Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Additional analysis plots
    plot_joint_analysis(dof_names, dof_positions, dof_velocities, time)


def plot_joint_analysis(dof_names, dof_positions, dof_velocities, time):
    """
    Create additional analysis plots for each joint.
    """
    
    num_dofs = len(dof_names)
    
    # Create individual joint plots
    fig, axes = plt.subplots(num_dofs, 2, figsize=(15, 4 * num_dofs))
    if num_dofs == 1:
        axes = axes.reshape(1, -1)
    
    for i, dof_name in enumerate(dof_names):
        # Position plot
        ax_pos = axes[i, 0]
        ax_pos.plot(time, dof_positions[:, i], 'b-', linewidth=2)
        ax_pos.set_xlabel('Time (s)')
        ax_pos.set_ylabel('Position (rad)')
        ax_pos.set_title(f'{dof_name} - Position')
        ax_pos.grid(True, alpha=0.3)
        
        # Add statistics
        pos_mean = np.mean(dof_positions[:, i])
        pos_std = np.std(dof_positions[:, i])
        pos_range = np.ptp(dof_positions[:, i])
        ax_pos.axhline(y=pos_mean, color='r', linestyle='--', alpha=0.7, label=f'Mean: {pos_mean:.3f}')
        ax_pos.text(0.02, 0.95, f'Range: {pos_range:.3f} rad\nStd: {pos_std:.3f} rad', 
                   transform=ax_pos.transAxes, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Velocity plot
        ax_vel = axes[i, 1]
        ax_vel.plot(time, dof_velocities[:, i], 'g-', linewidth=2)
        ax_vel.set_xlabel('Time (s)')
        ax_vel.set_ylabel('Velocity (rad/s)')
        ax_vel.set_title(f'{dof_name} - Velocity')
        ax_vel.grid(True, alpha=0.3)
        
        # Add statistics
        vel_mean = np.mean(dof_velocities[:, i])
        vel_std = np.std(dof_velocities[:, i])
        vel_max = np.max(np.abs(dof_velocities[:, i]))
        ax_vel.axhline(y=vel_mean, color='r', linestyle='--', alpha=0.7, label=f'Mean: {vel_mean:.3f}')
        ax_vel.text(0.02, 0.95, f'Max |vel|: {vel_max:.3f} rad/s\nStd: {vel_std:.3f} rad/s', 
                   transform=ax_vel.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


def plot_histograms(filename="ik_trajectory_data.npz", bins=50):
    """
    Create histograms for joint positions and velocities.
    
    Args:
        filename: Path to the NPZ file containing trajectory data
        bins: Number of histogram bins
    """
    
    data = np.load(filename)
    dof_names = data["dof_names"]
    dof_positions = data["dof_positions"]
    dof_velocities = data["dof_velocities"]
    
    num_dofs = len(dof_names)
    
    # Create figure with subplots for positions and velocities
    fig, axes = plt.subplots(num_dofs, 2, figsize=(15, 4 * num_dofs))
    if num_dofs == 1:
        axes = axes.reshape(1, -1)
    
    for i, dof_name in enumerate(dof_names):
        # Position histogram
        ax_pos = axes[i, 0]
        pos_data = dof_positions[:, i]
        n_pos, bins_pos, patches_pos = ax_pos.hist(pos_data, bins=bins, alpha=0.7, 
                                                  color='blue', edgecolor='black', linewidth=0.5)
        
        ax_pos.set_xlabel('Joint Position (rad)')
        ax_pos.set_ylabel('Frequency')
        ax_pos.set_title(f'{dof_name} - Position Distribution')
        ax_pos.grid(True, alpha=0.3)
        
        # Add statistics to position histogram
        pos_mean = np.mean(pos_data)
        pos_std = np.std(pos_data)
        pos_median = np.median(pos_data)
        ax_pos.axvline(pos_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {pos_mean:.3f}')
        ax_pos.axvline(pos_median, color='green', linestyle='--', linewidth=2, label=f'Median: {pos_median:.3f}')
        ax_pos.legend()
        
        # Add text box with statistics
        stats_text = f'Mean: {pos_mean:.3f} rad\nStd: {pos_std:.3f} rad\nMin: {np.min(pos_data):.3f} rad\nMax: {np.max(pos_data):.3f} rad'
        ax_pos.text(0.7, 0.95, stats_text, transform=ax_pos.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Velocity histogram
        ax_vel = axes[i, 1]
        vel_data = dof_velocities[:, i]
        n_vel, bins_vel, patches_vel = ax_vel.hist(vel_data, bins=bins, alpha=0.7, 
                                                  color='green', edgecolor='black', linewidth=0.5)
        
        ax_vel.set_xlabel('Joint Angular Velocity (rad/s)')
        ax_vel.set_ylabel('Frequency')
        ax_vel.set_title(f'{dof_name} - Velocity Distribution')
        ax_vel.grid(True, alpha=0.3)
        
        # Add statistics to velocity histogram
        vel_mean = np.mean(vel_data)
        vel_std = np.std(vel_data)
        vel_median = np.median(vel_data)
        ax_vel.axvline(vel_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {vel_mean:.3f}')
        ax_vel.axvline(vel_median, color='orange', linestyle='--', linewidth=2, label=f'Median: {vel_median:.3f}')
        ax_vel.legend()
        
        # Add text box with statistics
        stats_text = f'Mean: {vel_mean:.3f} rad/s\nStd: {vel_std:.3f} rad/s\nMin: {np.min(vel_data):.3f} rad/s\nMax: {np.max(vel_data):.3f} rad/s'
        ax_vel.text(0.7, 0.95, stats_text, transform=ax_vel.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


def plot_combined_histograms(filename="ik_trajectory_data.npz", bins=50):
    """
    Create combined histograms showing all joints together.
    
    Args:
        filename: Path to the NPZ file containing trajectory data
        bins: Number of histogram bins
    """
    
    data = np.load(filename)
    dof_names = data["dof_names"]
    dof_positions = data["dof_positions"]
    dof_velocities = data["dof_velocities"]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Combined position histogram
    for i, dof_name in enumerate(dof_names):
        ax1.hist(dof_positions[:, i], bins=bins, alpha=0.6, label=dof_name, density=True)
    
    ax1.set_xlabel('Joint Position (rad)')
    ax1.set_ylabel('Density')
    ax1.set_title('Combined Position Distributions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Combined velocity histogram
    for i, dof_name in enumerate(dof_names):
        ax2.hist(dof_velocities[:, i], bins=bins, alpha=0.6, label=dof_name, density=True)
    
    ax2.set_xlabel('Joint Angular Velocity (rad/s)')
    ax2.set_ylabel('Density')
    ax2.set_title('Combined Velocity Distributions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_2d_histograms(filename="ik_trajectory_data.npz", bins=50):
    """
    Create 2D histograms with joint position (x) vs joint angular velocity (y).
    
    Args:
        filename: Path to the NPZ file containing trajectory data
        bins: Number of histogram bins
    """
    
    data = np.load(filename)
    dof_names = data["dof_names"]
    dof_positions = data["dof_positions"]
    dof_velocities = data["dof_velocities"]
    
    num_dofs = len(dof_names)
    cols = min(3, num_dofs)
    rows = (num_dofs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if num_dofs == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, dof_name in enumerate(dof_names):
        ax = axes[i]
        
        # Create 2D histogram
        hist, xedges, yedges = np.histogram2d(dof_positions[:, i], dof_velocities[:, i], bins=bins)
        
        # Plot 2D histogram
        im = ax.imshow(hist.T, origin='lower', aspect='auto', cmap='viridis',
                      extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        
        ax.set_xlabel('Joint Position (rad)')
        ax.set_ylabel('Joint Angular Velocity (rad/s)')
        ax.set_title(f'{dof_name} - Position vs Velocity Distribution')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Frequency')
        
        # Add contour lines
        X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
        ax.contour(X, Y, hist.T, levels=5, colors='white', alpha=0.5, linewidths=0.5)
    
    # Hide unused subplots
    for i in range(num_dofs, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_position_spectrum(filename="ik_trajectory_data.npz"):
    """
    Create frequency spectrum analysis of joint positions.
    
    Args:
        filename: Path to the NPZ file containing trajectory data
    """
    
    data = np.load(filename)
    dof_names = data["dof_names"]
    dof_positions = data["dof_positions"]
    fps = data["fps"]
    
    num_dofs = len(dof_names)
    cols = min(3, num_dofs)
    rows = (num_dofs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if num_dofs == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, dof_name in enumerate(dof_names):
        ax = axes[i]
        
        # Compute FFT
        position_signal = dof_positions[:, i]
        # Remove DC component (mean)
        position_signal = position_signal - np.mean(position_signal)
        
        # Apply window function to reduce spectral leakage
        windowed_signal = position_signal * np.hanning(len(position_signal))
        
        # Compute FFT
        fft_result = np.fft.fft(windowed_signal)
        frequencies = np.fft.fftfreq(len(windowed_signal), 1/fps)
        
        # Take only positive frequencies
        positive_freq_idx = frequencies > 0
        frequencies = frequencies[positive_freq_idx]
        magnitude = np.abs(fft_result[positive_freq_idx])
        
        # Convert to dB scale
        magnitude_db = 20 * np.log10(magnitude + 1e-12)  # Add small value to avoid log(0)
        
        # Plot spectrum
        ax.plot(frequencies, magnitude_db, 'b-', linewidth=1)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_title(f'{dof_name} - Position Frequency Spectrum')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, fps/2)  # Nyquist frequency
        
        # Find and mark dominant frequencies
        peak_indices = np.argsort(magnitude)[-3:]  # Top 3 peaks
        for peak_idx in peak_indices:
            if magnitude[peak_idx] > np.max(magnitude) * 0.1:  # Only show significant peaks
                ax.axvline(frequencies[peak_idx], color='red', linestyle='--', alpha=0.7)
                ax.text(frequencies[peak_idx], magnitude_db[peak_idx] + 5, 
                       f'{frequencies[peak_idx]:.2f} Hz', 
                       rotation=90, ha='center', va='bottom', fontsize=8)
    
    # Hide unused subplots
    for i in range(num_dofs, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_spectrogram(filename="ik_trajectory_data.npz"):
    """
    Create spectrograms showing how the frequency content changes over time.
    
    Args:
        filename: Path to the NPZ file containing trajectory data
    """
    
    data = np.load(filename)
    dof_names = data["dof_names"]
    dof_positions = data["dof_positions"]
    fps = data["fps"]
    
    num_dofs = len(dof_names)
    cols = min(2, num_dofs)
    rows = (num_dofs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 4 * rows))
    if num_dofs == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, dof_name in enumerate(dof_names):
        ax = axes[i]
        
        # Compute spectrogram
        position_signal = dof_positions[:, i]
        position_signal = position_signal - np.mean(position_signal)
        
        # Parameters for spectrogram
        nperseg = min(256, len(position_signal) // 4)  # Window size
        noverlap = nperseg // 2  # 50% overlap
        
        from scipy import signal
        frequencies, times, Sxx = signal.spectrogram(position_signal, fps, 
                                                    nperseg=nperseg, noverlap=noverlap)
        
        # Convert to dB scale
        Sxx_db = 10 * np.log10(Sxx + 1e-12)
        
        # Plot spectrogram
        im = ax.pcolormesh(times, frequencies, Sxx_db, shading='gouraud', cmap='viridis')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'{dof_name} - Position Spectrogram')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power Spectral Density (dB/Hz)')
    
    # Hide unused subplots
    for i in range(num_dofs, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_phase_portraits(filename="ik_trajectory_data.npz"):
    """
    Create phase portraits (position vs velocity) for each joint.
    """
    
    data = np.load(filename)
    dof_names = data["dof_names"]
    dof_positions = data["dof_positions"]
    dof_velocities = data["dof_velocities"]
    
    num_dofs = len(dof_names)
    cols = min(3, num_dofs)
    rows = (num_dofs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if num_dofs == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, dof_name in enumerate(dof_names):
        ax = axes[i]
        
        # Create phase portrait
        scatter = ax.scatter(dof_positions[:, i], dof_velocities[:, i], 
                           c=np.arange(len(dof_positions[:, i])), 
                           cmap='viridis', s=10, alpha=0.7)
        
        ax.set_xlabel('Position (rad)')
        ax.set_ylabel('Velocity (rad/s)')
        ax.set_title(f'{dof_name} - Phase Portrait')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for time
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Time Step')
    
    # Hide unused subplots
    for i in range(num_dofs, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def print_statistics(filename="ik_trajectory_data.npz"):
    """
    Print detailed statistics about the trajectory data.
    """
    
    data = np.load(filename)
    dof_names = data["dof_names"]
    dof_positions = data["dof_positions"]
    dof_velocities = data["dof_velocities"]
    fps = data["fps"]
    
    print("\n" + "="*60)
    print("TRAJECTORY DATA STATISTICS")
    print("="*60)
    
    print(f"Duration: {dof_positions.shape[0] / fps:.3f} seconds")
    print(f"Sample rate: {fps} Hz")
    print(f"Number of samples: {dof_positions.shape[0]}")
    print(f"Number of DOFs: {len(dof_names)}")
    
    print("\nJOINT STATISTICS:")
    print("-" * 60)
    print(f"{'Joint Name':<15} {'Pos Mean':<10} {'Pos Std':<10} {'Pos Range':<12} {'Vel Max':<10}")
    print("-" * 60)
    
    for i, dof_name in enumerate(dof_names):
        pos_mean = np.mean(dof_positions[:, i])
        pos_std = np.std(dof_positions[:, i])
        pos_range = np.ptp(dof_positions[:, i])
        vel_max = np.max(np.abs(dof_velocities[:, i]))
        
        print(f"{dof_name:<15} {pos_mean:<10.4f} {pos_std:<10.4f} {pos_range:<12.4f} {vel_max:<10.4f}")


def export_to_csv(filename="ik_trajectory_data.npz", output_csv="joint_data.csv"):
    """
    Export joint data to CSV file with specified format.
    
    Args:
        filename: Path to the NPZ file containing trajectory data
        output_csv: Output CSV filename
    """
    
    # Load data
    data = np.load(filename)
    fps = data["fps"]
    dof_names = data["dof_names"]
    dof_positions = data["dof_positions"]
    
    # Create time array
    num_timesteps = dof_positions.shape[0]
    time = np.arange(num_timesteps) / fps
    
    # Define the joints we want to export
    target_joints = ['L_hip_joint', 'L_hip2_joint', 'L_thigh_joint', 'L_calf_joint', 'L_toe_joint']
    
    # Create a dictionary to store the data
    csv_data = {'timestamp': time}
    
    # Find the indices of the target joints and add their data
    for joint_name in target_joints:
        if joint_name in dof_names:
            joint_idx = list(dof_names).index(joint_name)
            csv_data[joint_name] = dof_positions[:, joint_idx]
        else:
            print(f"Warning: Joint '{joint_name}' not found in data. Available joints: {list(dof_names)}")
            # Add a column of zeros if joint not found
            csv_data[joint_name] = np.zeros(len(time))
    
    # Create DataFrame and export to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(output_csv, index=False, float_format='%.6f')
    
    print(f"Data exported to {output_csv}")
    print(f"Exported {len(df)} timesteps with columns: {list(df.columns)}")
    print(f"Time range: {time[0]:.3f} to {time[-1]:.3f} seconds")
    
    return df


def main():
    """Main function with command line argument parsing."""
    
    parser = argparse.ArgumentParser(description="Plot IK trajectory data from NPZ file")
    parser.add_argument("--file", "-f", type=str, default="ik_trajectory_data_2.npz",
                       help="Path to the NPZ file (default: ik_trajectory_data.npz)")
    parser.add_argument("--stats", "-s", action="store_true",
                       help="Print detailed statistics")
    parser.add_argument("--phase", "-p", action="store_true",
                       help="Show phase portraits")
    parser.add_argument("--hist", action="store_true",
                       help="Show histograms of joint positions and velocities")
    parser.add_argument("--combined-hist", action="store_true",
                       help="Show combined histograms of all joints")
    parser.add_argument("--hist2d", action="store_true",
                       help="Show 2D histograms (position vs velocity)")
    parser.add_argument("--spectrum", action="store_true",
                       help="Show frequency spectrum of joint positions")
    parser.add_argument("--spectrogram", action="store_true",
                       help="Show spectrograms of joint positions")
    parser.add_argument("--hist3d", action="store_true",
                       help="Show 3D bar histograms of joint positions vs velocities")
    parser.add_argument("--surface3d", action="store_true",
                       help="Show 3D surface histograms of joint positions vs velocities")
    parser.add_argument("--export-csv", type=str, metavar="OUTPUT_FILE",
                       help="Export joint data to CSV file (specify output filename)")
    parser.add_argument("--bins", type=int, default=50,
                       help="Number of bins for histograms (default: 50)")
    parser.add_argument("--all", "-a", action="store_true",
                       help="Show all plots and statistics")
    
    args = parser.parse_args()
    
    filename = args.file
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found!")
        print("Make sure to run the IK trajectory script first to generate the data.")
        return
    
    # Show main plots
    plot_trajectory_data(filename)
    
    # Show additional analysis based on arguments
    if args.stats or args.all:
        print_statistics(filename)
    
    if args.phase or args.all:
        plot_phase_portraits(filename)
    
    if args.hist or args.all:
        plot_histograms(filename, bins=args.bins)
    
    if args.combined_hist or args.all:
        plot_combined_histograms(filename, bins=args.bins)
    
    if args.hist2d or args.all:
        plot_2d_histograms(filename, bins=args.bins)
    
    if args.spectrum or args.all:
        plot_position_spectrum(filename)
    
    if args.spectrogram or args.all:
        plot_spectrogram(filename)
    
    if args.hist3d or args.all:
        plot_3d_histogram(filename, bins=args.bins)
    
    if args.surface3d or args.all:
        plot_3d_surface_histogram(filename, bins=args.bins)
    
    # Export to CSV if requested
    if args.export_csv:
        export_to_csv(filename, args.export_csv)


if __name__ == "__main__":
    main()