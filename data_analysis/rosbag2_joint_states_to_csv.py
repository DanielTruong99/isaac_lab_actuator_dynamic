import sys
import os
import csv
from typing import List, Optional

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
import rclpy.serialization as rserialize
from sensor_msgs.msg import JointState


def resolve_bag_uri(path: str) -> str:
    path = os.path.abspath(path)
    if os.path.isdir(path):
        meta = os.path.join(path, 'metadata.yaml')
        if os.path.exists(meta):
            return path
        return path
    if path.endswith('.db3') and os.path.isfile(path):
        return os.path.dirname(path)
    return os.path.dirname(path)


def write_jointstate_rows(writer: csv.DictWriter, topic: str, bag_time_ns: int, msg: JointState):
    # Prefer header stamp when available
    stamp_ns = (msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec) if msg.header else 0
    stamp_ns = stamp_ns or bag_time_ns
    stamp_sec = stamp_ns / 1e9

    names: List[str] = list(msg.name)
    positions: List[Optional[float]] = list(msg.position)
    velocities: List[Optional[float]] = list(msg.velocity)
    efforts: List[Optional[float]] = list(msg.effort)

    n = len(names)
    # Ensure arrays are at least n long; fill missing with None
    def get(arr, i):
        return arr[i] if i < len(arr) else None

    for i in range(n):
        writer.writerow({
            'topic': topic,
            'stamp_sec': int(stamp_ns // 1_000_000_000),
            'stamp_nanosec': int(stamp_ns % 1_000_000_000),
            'time_ns': bag_time_ns,
            'time_sec': bag_time_ns / 1e9,
            'joint': names[i],
            'position': get(positions, i),
            'velocity': get(velocities, i),
            'effort': get(efforts, i),
        })


def main(input_path: str, output_csv: str):
    uri = resolve_bag_uri(input_path)

    storage_options = StorageOptions(uri=uri, storage_id='sqlite3')
    converter_options = ConverterOptions('', '')

    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    topics_types = reader.get_all_topics_and_types()
    joint_topics = {t.name for t in topics_types if t.type in ('sensor_msgs/msg/JointState', 'sensor_msgs/JointState')}
    if not joint_topics:
        print('No JointState topics found in bag. Topics present:')
        for t in topics_types:
            print(f"- {t.name}: {t.type}")
        sys.exit(1)

    header = ['topic', 'stamp_sec', 'stamp_nanosec', 'time_ns', 'time_sec', 'joint', 'position', 'velocity', 'effort']
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)) or '.', exist_ok=True)
    rows = 0
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        while reader.has_next():
            topic, data, t = reader.read_next()
            if topic not in joint_topics:
                continue
            msg = rserialize.deserialize_message(data, JointState)
            write_jointstate_rows(writer, topic, t, msg)
            # Estimate rows increment by number of joints
            rows += max(1, len(msg.name))

    print(f"Exported JointState rows: {rows} to {output_csv}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python3 rosbag2_joint_states_to_csv.py <rosbag_dir_or_db3_file> <output_csv>')
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
