#!/usr/bin/env python3
"""
Decode raw rosbag2 CSV rows (base64) for vectornav_msgs/msg/CommonGroup into a readable CSV.

Input CSV can be either the combined raw CSV from rosbag2_csv.py (--raw --combined) with columns:
  topic,type,serialization_format,time_ns,time_sec,data_base64
or a per-topic raw CSV with columns:
  time_ns,time_sec,data_base64,type,serialization_format

Requires ROS 2 Python libraries in the environment:
  - rclpy
  - rosidl_runtime_py
  - vectornav_msgs installed

Usage:
  python3 decode_common_group_csv.py --input <raw_csv> --out <decoded_csv>
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
from typing import Any, Dict, List


def flatten(value: Any, prefix: str = 'data') -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    def _rec(v: Any, key: str):
        if isinstance(v, dict):
            for k, vv in v.items():
                _rec(vv, f"{key}.{k}" if key else str(k))
        elif isinstance(v, (list, tuple)):
            try:
                out[key] = json.dumps(v)
            except Exception:
                out[key] = str(v)
        elif isinstance(v, (bytes, bytearray)):
            out[key] = base64.b64encode(bytes(v)).decode('ascii')
        else:
            out[key] = v

    _rec(value, prefix)
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Decode CommonGroup raw CSV to readable CSV')
    p.add_argument('--input', required=True, help='Path to raw CSV (combined or per-topic)')
    p.add_argument('--out', required=True, help='Path to output decoded CSV')
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        import rclpy.serialization as rserialize  # type: ignore
        from rosidl_runtime_py.utilities import get_message  # type: ignore
        from rosidl_runtime_py import message_to_ordereddict  # type: ignore
    except Exception as e:
        print('ROS 2 Python libraries not available. Please source your ROS 2 environment and try again.')
        print('Error:', e)
        return 1

    msg_type = 'vectornav_msgs/msg/CommonGroup'
    try:
        py_cls = get_message(msg_type)
    except Exception as e:
        print(f'Cannot load message type {msg_type}. Ensure vectornav_msgs is installed. Error: {e}')
        return 1

    rows: List[Dict[str, Any]] = []
    header_superset: List[str] = []

    with open(args.input, 'r') as f:
        reader = csv.DictReader(f)
        combined_cols = {'topic', 'type', 'serialization_format', 'time_ns', 'time_sec', 'data_base64'}
        per_topic_cols = {'time_ns', 'time_sec', 'data_base64', 'type', 'serialization_format'}
        cols = set(reader.fieldnames or [])
        mode = 'combined' if combined_cols.issubset(cols) else ('per-topic' if per_topic_cols.issubset(cols) else 'unknown')
        if mode == 'unknown':
            print('Unrecognized CSV header for raw data. Got columns:', reader.fieldnames)
            return 1
        for r in reader:
            if r.get('type') != msg_type:
                continue
            data_b64 = r.get('data_base64') or ''
            if not data_b64:
                continue
            try:
                payload = base64.b64decode(data_b64)
                msg = rserialize.deserialize_message(payload, py_cls)
                md = message_to_ordereddict(msg)
            except Exception as e:
                print('Failed to decode a row; skipping. Error:', e)
                continue

            stamp_ns = 0
            try:
                h = md.get('header', {}) if isinstance(md, dict) else {}
                sec = int(h.get('stamp', {}).get('sec', 0))
                nsec = int(h.get('stamp', {}).get('nanosec', 0))
                stamp_ns = sec * 1_000_000_000 + nsec
            except Exception:
                pass
            t_ns = int(r.get('time_ns', 0))
            row = {
                'stamp_sec': int(stamp_ns // 1_000_000_000),
                'stamp_nanosec': int(stamp_ns % 1_000_000_000),
                'bag_time_ns': t_ns,
                'bag_time_sec': float(r.get('time_sec', 0.0) or 0.0),
            }
            row.update(flatten(md, prefix='data'))
            # track header superset
            for k in row.keys():
                if k not in header_superset:
                    header_superset.append(k)
            rows.append(row)

    if not rows:
        print('No CommonGroup rows decoded; nothing to write.')
        return 0

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header_superset)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f'Wrote decoded CSV with {len(rows)} rows to {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
