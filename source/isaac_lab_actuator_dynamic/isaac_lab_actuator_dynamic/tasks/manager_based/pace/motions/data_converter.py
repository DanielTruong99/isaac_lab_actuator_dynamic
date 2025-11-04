import numpy as np
import pandas as pd
import os

motion_file = "source/isaac_lab_actuator_dynamic/isaac_lab_actuator_dynamic/tasks/direct/actuator_dynamic/motions/recorded_left_leg.csv"
motion_data: pd.DataFrame = pd.read_csv(motion_file)

# get joint position
L_hip2_joint_position_index = motion_data["topic"] == "/joint_states.position[1]"
dof_positions = motion_data["value"][L_hip2_joint_position_index].to_numpy()
time_stamps_positions = motion_data["timestamp"][L_hip2_joint_position_index].to_numpy()

# get joint velocity
L_hip2_joint_velocity_index = motion_data["topic"] == "/joint_states.velocity[1]"
dof_velocities = motion_data["value"][L_hip2_joint_velocity_index].to_numpy()
time_stamps_velocities = motion_data["timestamp"][L_hip2_joint_velocity_index].to_numpy()

# get joint torque
L_hip2_joint_torque_index = motion_data["topic"] == "/joint_states.effort[1]"
dof_efforts = motion_data["value"][L_hip2_joint_torque_index].to_numpy()
time_stamps_efforts = motion_data["timestamp"][L_hip2_joint_torque_index].to_numpy()
print(time_stamps_efforts[0])

# get joint position command
L_hip2_joint_position_command_index = motion_data["topic"] == "/joint_cmds.position[1]"
dof_position_commands = motion_data["value"][L_hip2_joint_position_command_index].to_numpy()
time_stamps_position_commands = motion_data["timestamp"][L_hip2_joint_position_command_index].to_numpy()


# Save the data to a compressed .npz file
time_stamp_range =[1760511144.0, 1760511150.19]
data_dict = {
    "fps": np.array(200),
    "dof_names": np.array(["L_hip2_joint"]),
    "dof_positions": dof_positions[(time_stamps_positions >= time_stamp_range[0]) & (time_stamps_positions <= time_stamp_range[1])],
    "dof_velocities": dof_velocities[(time_stamps_velocities >= time_stamp_range[0]) & (time_stamps_velocities <= time_stamp_range[1])],
    "dof_efforts": dof_efforts[(time_stamps_efforts >= time_stamp_range[0]) & (time_stamps_efforts <= time_stamp_range[1])],
    "dof_position_commands": dof_position_commands[(time_stamps_position_commands >= time_stamp_range[0]) & (time_stamps_position_commands <= time_stamp_range[1])],
}

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
time_position_cmd = time_stamps_position_commands[(time_stamps_position_commands >= time_stamp_range[0]) & (time_stamps_position_commands <= time_stamp_range[1])]
time_position = time_stamps_positions[(time_stamps_positions >= time_stamp_range[0]) & (time_stamps_positions <= time_stamp_range[1])]
plt.plot(time_position, data_dict["dof_positions"], label='Joint Position')
plt.plot(time_position_cmd, data_dict["dof_position_commands"], label='Joint Position Command',
          linestyle='--')
plt.title('Joint Position')
plt.ylabel('Position (rad)')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(data_dict["dof_velocities"], label='Joint Velocity', color='orange')
plt.title('Joint Velocity')
plt.ylabel('Velocity (rad/s)')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(data_dict["dof_efforts"], label='Joint Effort', color='green')
plt.title('Joint Effort')
plt.ylabel('Effort (Nm)')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(data_dict["dof_position_commands"], label='Joint Position Command', color='red')
plt.title('Joint Position Command')
plt.ylabel('Position Command (rad)')
plt.legend()

plt.tight_layout()
plt.show()

current_path = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(current_path, "recorded_motor_data_1.npz")
np.savez_compressed(output_file, **data_dict)

# Save the data to a compressed .npz file
time_stamp_range =[1748497661.8, 1748497671.78]
data_dict = {
    "fps": np.array(200),
    "dof_names": np.array(["L_hip2_joint"]),
    "dof_positions": dof_positions[(time_stamps_positions >= time_stamp_range[0]) & (time_stamps_positions <= time_stamp_range[1])],
    "dof_velocities": dof_velocities[(time_stamps_velocities >= time_stamp_range[0]) & (time_stamps_velocities <= time_stamp_range[1])],
    "dof_efforts": dof_efforts[(time_stamps_efforts >= time_stamp_range[0]) & (time_stamps_efforts <= time_stamp_range[1])],
    "dof_position_commands": dof_position_commands[(time_stamps_position_commands >= time_stamp_range[0]) & (time_stamps_position_commands <= time_stamp_range[1])],
}

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
time_position_cmd = time_stamps_position_commands[(time_stamps_position_commands >= time_stamp_range[0]) & (time_stamps_position_commands <= time_stamp_range[1])]
time_position = time_stamps_positions[(time_stamps_positions >= time_stamp_range[0]) & (time_stamps_positions <= time_stamp_range[1])]
plt.plot(time_position, data_dict["dof_positions"], label='Joint Position')
plt.plot(time_position_cmd, data_dict["dof_position_commands"], label='Joint Position Command', linestyle='--')
plt.title('Joint Position')
plt.ylabel('Position (rad)')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(data_dict["dof_velocities"], label='Joint Velocity', color='orange')
plt.title('Joint Velocity')
plt.ylabel('Velocity (rad/s)')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(data_dict["dof_efforts"], label='Joint Effort', color='green')
plt.title('Joint Effort')
plt.ylabel('Effort (Nm)')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(data_dict["dof_position_commands"], label='Joint Position Command', color='red')
plt.title('Joint Position Command')
plt.ylabel('Position Command (rad)')
plt.legend()

plt.tight_layout()
plt.show()

current_path = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(current_path, "recorded_motor_data_2.npz")
np.savez_compressed(output_file, **data_dict)

