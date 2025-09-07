import sys
import os
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


LEFT_JOINTS: List[str] = [
    'L_hip_joint',
    'L_hip2_joint',
    'L_thigh_joint',
    'L_calf_joint',
    'L_toe_joint',
]

# PID gains for each joint (from your system configuration)
JOINT_GAINS = {
    'L_hip_joint': {'kp': 50.0, 'kd': 3.0},      # Hip roll
    'L_hip2_joint': {'kp': 70.0, 'kd': 4.0},     # Hip roll
    'L_thigh_joint': {'kp': 270.0, 'kd': 4.0},   # Hip pitch
    'L_calf_joint': {'kp': 120.0, 'kd': 1.0},    # Knee pitch
    'L_toe_joint': {'kp': 50.0, 'kd': 1.0},      # Ankle pitch
}


def _load_and_validate(csv_path: str, normalize_time: bool = True) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"topic", "time_sec", "joint", "position"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    
    if normalize_time:
        try:
            t0 = float(df['time_sec'].min())
            if np.isfinite(t0):
                df['time_sec'] = df['time_sec'] - t0
        except Exception:
            pass
    return df


def compute_pd_torque(position_cmd, position_actual, velocity_actual, kp=100.0, kd=10.0):
    """
    Compute PD control torque: tau = kp * (pos_cmd - pos_actual) - kd * vel_actual
    """
    position_error = position_cmd - position_actual
    torque = kp * position_error - kd * velocity_actual
    return torque


def interpolate_commands(cmd_df, actual_df, time_col='time_sec'):
    """
    Interpolate command positions to match actual joint state timestamps
    """
    # Remove any NaN or infinite values
    cmd_df_clean = cmd_df.dropna(subset=['position', time_col])
    actual_df_clean = actual_df.dropna(subset=[time_col])
    
    if len(cmd_df_clean) == 0 or len(actual_df_clean) == 0:
        return np.array([])
    
    # Interpolate command positions at actual timestamps
    cmd_positions_interp = np.interp(
        actual_df_clean[time_col], 
        cmd_df_clean[time_col], 
        cmd_df_clean['position']
    )
    
    return cmd_positions_interp, actual_df_clean


def analyze_joint(data_frame: pd.DataFrame, joint_name: str):
    """
    Analyze a specific joint: compare command effort with computed PD torque
    """
    print(f"\nAnalyzing joint: {joint_name}")
    
    # Get gains for this joint
    if joint_name not in JOINT_GAINS:
        print(f"No gains configured for {joint_name}")
        return None
    
    kp = JOINT_GAINS[joint_name]['kp']
    kd = JOINT_GAINS[joint_name]['kd']
    print(f"Using gains: Kp={kp}, Kd={kd}")
    
    # Get actual joint states (with effort)
    actual_cols = ['time_sec', 'position', 'velocity'] + (['effort'] if 'effort' in data_frame.columns else [])
    actual_df = data_frame[
        (data_frame['topic'] == '/joint_states') & 
        (data_frame['joint'] == joint_name)
    ][actual_cols].copy()
    actual_df = actual_df.sort_values('time_sec')
    
    # Get command positions from filtered_joint_cmd
    cmd_df = data_frame[
        (data_frame['topic'] == '/filtered_joint_cmd') & 
        (data_frame['joint'] == joint_name)
    ][['time_sec', 'position']].copy()
    cmd_df = cmd_df.sort_values('time_sec')
    
    print(f"Found {len(actual_df)} actual joint state samples")
    print(f"Found {len(cmd_df)} command samples")
    
    if len(actual_df) == 0 or len(cmd_df) == 0:
        print(f"No data found for {joint_name}")
        return None
    
    # Interpolate command positions to match actual timestamps
    try:
        cmd_positions_interp, actual_df_clean = interpolate_commands(cmd_df, actual_df)
        
        if len(cmd_positions_interp) == 0:
            print(f"Failed to interpolate commands for {joint_name}")
            return None
            
    except Exception as e:
        print(f"Error interpolating commands for {joint_name}: {e}")
        return None
    
    # Compute PD torque
    pd_torque = compute_pd_torque(
        cmd_positions_interp,
        actual_df_clean['position'].values,
        actual_df_clean['velocity'].values,
        kp=kp,
        kd=kd
    )
    
    # Create result dataframe
    result_df = actual_df_clean.copy()
    result_df['position_cmd'] = cmd_positions_interp
    result_df['pd_torque'] = pd_torque
    result_df['position_error'] = cmd_positions_interp - result_df['position']
    result_df['kp'] = kp
    result_df['kd'] = kd
    
    return result_df


def plot_comparison(result_df: pd.DataFrame, joint_name: str):
    """
    Plot comparison between command effort and computed PD torque
    """
    if result_df is None or len(result_df) == 0:
        print(f"No data to plot for {joint_name}")
        return
    
    kp = result_df['kp'].iloc[0]
    kd = result_df['kd'].iloc[0]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Command Effort vs PD Torque Comparison - {joint_name}\n(Kp={kp}, Kd={kd})', fontsize=14)
    
    # Plot 1: Position tracking
    axes[0, 0].plot(result_df['time_sec'], result_df['position'], label='Actual Position', alpha=0.8)
    axes[0, 0].plot(result_df['time_sec'], result_df['position_cmd'], label='Command Position', alpha=0.8)
    axes[0, 0].set_ylabel('Position (rad)')
    axes[0, 0].set_title('Position Tracking')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Position error
    axes[0, 1].plot(result_df['time_sec'], result_df['position_error'], color='red', alpha=0.8)
    axes[0, 1].set_ylabel('Position Error (rad)')
    axes[0, 1].set_title('Position Error')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Torque comparison
    if 'effort' in result_df.columns:
        axes[1, 0].plot(result_df['time_sec'], result_df['effort'], label='Command Effort', alpha=0.8)
    axes[1, 0].plot(result_df['time_sec'], result_df['pd_torque'], label='Computed PD Torque', alpha=0.8)
    axes[1, 0].set_ylabel('Torque (Nm)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_title('Torque Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Torque difference (if command effort available)
    if 'effort' in result_df.columns:
        torque_diff = result_df['effort'] - result_df['pd_torque']
        axes[1, 1].plot(result_df['time_sec'], torque_diff, color='purple', alpha=0.8)
        axes[1, 1].set_ylabel('Torque Difference (Nm)')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_title('Command Effort - PD Torque')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Print statistics
        print(f"\nTorque difference statistics for {joint_name}:")
        print(f"  Mean difference: {np.mean(torque_diff):.4f} Nm")
        print(f"  RMS difference: {np.sqrt(np.mean(torque_diff**2)):.4f} Nm")
        print(f"  Max absolute difference: {np.max(np.abs(torque_diff)):.4f} Nm")
    else:
        axes[1, 1].text(0.5, 0.5, 'No effort data\navailable', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Command Effort Not Available')
    
    plt.tight_layout()
    plt.show()


def main():
    # Load data
    csv_path = "data_analysis/joint_state_0905_rlpolicy.csv"
    print(f"Loading data from {csv_path}...")
    data_frame = _load_and_validate(csv_path)
    
    # Analyze all left joints with their respective gains
    joint_name = 'L_hip_joint'  # Start with one joint, you can change this
    
    print(f"\nAnalyzing joint: {joint_name}")
    result_df = analyze_joint(data_frame, joint_name)
    
    if result_df is not None:
        plot_comparison(result_df, joint_name)
    
    # Uncomment below to analyze all left joints
    # for joint_name in LEFT_JOINTS:
    #     result_df = analyze_joint(data_frame, joint_name)
    #     if result_df is not None:
    #         plot_comparison(result_df, joint_name)


if __name__ == "__main__":
    main()
