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


def _load_and_validate(csv_path: str, normalize_time: bool = True) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"topic", "time_sec", "joint", "position"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    # effort is optional but used if available
    if normalize_time:
        try:
            t0 = float(df['time_sec'].min())
            if np.isfinite(t0):
                df['time_sec'] = df['time_sec'] - t0
        except Exception:
            pass
    return df




csv_path = "joint_state_latest_3.csv"
data_frame = _load_and_validate(csv_path)
joint_name = 'L_calf_joint'; cmd_topic = '/joint_states'

cmd_cols = ['time_sec', 'position'] + ([ 'effort' ] if 'effort' in data_frame.columns else [])
cmd_df = data_frame[(data_frame['topic'] == cmd_topic) & (data_frame['joint'] == joint_name)][cmd_cols].copy() # Get command data topic
cmd_df = cmd_df.sort_values('time_sec') # Sort by time

# Plot
fig, ax_pos = plt.subplots(figsize=(12, 6))
ax_pos.plot(cmd_df['time_sec'], cmd_df['effort'], label='Effort')
ax_pos.set_xlabel('Time (s)')
ax_pos.set_ylabel('Effort')
ax_pos.set_title(f"Effort for {joint_name} on {cmd_topic}")
ax_pos.legend()
plt.tight_layout()
plt.show()

