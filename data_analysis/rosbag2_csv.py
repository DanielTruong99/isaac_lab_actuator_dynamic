#!/usr/bin/env python3
"""
rosbag2_csv.py

Convert ROS 2 rosbag2 data (db3 + metadata.yaml) to CSV.

Two modes:
- Rich mode (default): uses rosbag2_py + rclpy + rosidl_runtime_py to deserialize
  messages to Python dicts and writes a per-topic CSV with flattened fields.
- Raw mode (fallback or --raw): uses sqlite3 and metadata.yaml to export timestamps
  and raw serialized message bytes (base64) without decoding the message schema.

Usage:
  python3 rosbag2_csv.py <bag_dir_or_db3> --out <output_dir> [--topics /a /b] [--raw] [--combined]

Notes:
- By default, writes one CSV per topic to <output_dir>. Use --combined to write a
  single combined CSV (adds a 'topic' column). Combined mode is supported only in
  raw mode for now; rich mode writes per-topic files to keep headers stable.
- Arrays and nested fields are JSON-encoded strings in the CSV to keep a stable
  header across messages.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


def resolve_bag_uri(path: str) -> str:
	path = os.path.abspath(path)
	if os.path.isdir(path):
		return path
	if path.endswith('.db3') and os.path.isfile(path):
		return os.path.dirname(path)
	return os.path.dirname(path)


def sanitize_topic(topic: str) -> str:
	# Make a filesystem-friendly filename
	name = topic.strip('/').replace('/', '__') or 'root'
	# Avoid awkward characters
	return ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in name)


def find_db3_files(bag_dir: str, metadata: Optional[dict] = None) -> List[str]:
	# Prefer files listed in metadata.yaml
	if metadata:
		rels = metadata.get('rosbag2_bagfile_information', {}).get('relative_file_paths', [])
		files = [os.path.join(bag_dir, r) for r in rels]
		return [f for f in files if os.path.exists(f)]
	# Fallback: list *.db3 in directory
	return sorted(
		[os.path.join(bag_dir, f) for f in os.listdir(bag_dir) if f.endswith('.db3')]
	)


def load_metadata_yaml(bag_dir: str) -> Optional[dict]:
	meta_path = os.path.join(bag_dir, 'metadata.yaml')
	if not os.path.exists(meta_path):
		return None
	try:
		import yaml  # type: ignore
	except Exception:
		return None
	with open(meta_path, 'r') as f:
		return yaml.safe_load(f)


def flatten(value: Any, prefix: str = 'data') -> Dict[str, Any]:
	"""Flatten nested dicts/lists to dotted keys. Arrays are JSON strings."""
	out: Dict[str, Any] = {}

	def _rec(v: Any, key: str):
		if isinstance(v, dict):
			for k, vv in v.items():
				_rec(vv, f"{key}.{k}" if key else str(k))
		elif isinstance(v, (list, tuple)):
			# encode arrays as JSON string to keep stable header
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


# -------- Rich mode using rosbag2_py (if available) --------

def try_import_rich_stack():
	try:
		from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions  # type: ignore
		import rclpy.serialization as rserialize  # type: ignore
		from rosidl_runtime_py.utilities import get_message  # type: ignore
		from rosidl_runtime_py import message_to_ordereddict  # type: ignore
		return {
			'SequentialReader': SequentialReader,
			'StorageOptions': StorageOptions,
			'ConverterOptions': ConverterOptions,
			'rserialize': rserialize,
			'get_message': get_message,
			'message_to_ordereddict': message_to_ordereddict,
		}
	except Exception:
		return None


@dataclass
class TopicWriter:
	file: Any
	writer: csv.DictWriter
	header: List[str]


def ensure_topic_writer(base_out_dir: str, topic: str, first_row: Dict[str, Any]) -> TopicWriter:
	os.makedirs(base_out_dir, exist_ok=True)
	filename = os.path.join(base_out_dir, f"{sanitize_topic(topic)}.csv")
	# Ensure deterministic header order
	header = list(first_row.keys())
	f = open(filename, 'w', newline='')
	writer = csv.DictWriter(f, fieldnames=header)
	writer.writeheader()
	return TopicWriter(file=f, writer=writer, header=header)


def rich_export(bag_dir: str, out_dir: str, topics: Optional[List[str]] = None) -> None:
	stack = try_import_rich_stack()
	if not stack:
		raise RuntimeError('rich stack not available')

	SequentialReader = stack['SequentialReader']
	StorageOptions = stack['StorageOptions']
	ConverterOptions = stack['ConverterOptions']
	rserialize = stack['rserialize']
	get_message = stack['get_message']
	message_to_ordereddict = stack['message_to_ordereddict']

	storage_options = StorageOptions(uri=bag_dir, storage_id='sqlite3')
	converter_options = ConverterOptions('', '')
	reader = SequentialReader()
	reader.open(storage_options, converter_options)

	topics_types = reader.get_all_topics_and_types()
	topic_to_type = {t.name: t.type for t in topics_types}
	selected = set(topics) if topics else set(topic_to_type.keys())
	invalid = [t for t in selected if t not in topic_to_type]
	if invalid:
		print(f"Warning: topics not found in bag: {invalid}")
		selected = selected.intersection(topic_to_type.keys())
	if not selected:
		print("No topics selected; nothing to export.")
		return

	# Prepare per-topic writers lazily
	writers: Dict[str, TopicWriter] = {}
	type_cache: Dict[str, Any] = {}

	count_by_topic: Dict[str, int] = {t: 0 for t in selected}

	try:
		while reader.has_next():
			topic, data, t_ns = reader.read_next()
			if topic not in selected:
				continue
			msg_type = topic_to_type[topic]
			if msg_type not in type_cache:
				try:
					type_cache[msg_type] = get_message(msg_type)
				except Exception as e:
					print(f"Failed to load message type {msg_type} for {topic}: {e}. Skipping.")
					selected.discard(topic)
					continue
			py_cls = type_cache[msg_type]
			try:
				msg = rserialize.deserialize_message(data, py_cls)
			except Exception as e:
				print(f"Failed to deserialize {topic} ({msg_type}): {e}. Skipping.")
				selected.discard(topic)
				continue

			# Convert to ordered dict and flatten arrays as JSON strings
			try:
				msg_dict = message_to_ordereddict(msg)
			except Exception:
				# graceful fallback to __dict__ if possible
				msg_dict = getattr(msg, '__dict__', {})

			# Prefer header timestamp if present
			header = msg_dict.get('header') if isinstance(msg_dict, dict) else None
			if isinstance(header, dict):
				sec = int(header.get('stamp', {}).get('sec', 0))
				nsec = int(header.get('stamp', {}).get('nanosec', 0))
				stamp_ns = sec * 1_000_000_000 + nsec
			else:
				stamp_ns = 0
			stamp_ns = stamp_ns or t_ns

			row: Dict[str, Any] = {
				'stamp_sec': int(stamp_ns // 1_000_000_000),
				'stamp_nanosec': int(stamp_ns % 1_000_000_000),
				'bag_time_ns': int(t_ns),
				'bag_time_sec': float(t_ns) / 1e9,
			}
			# Flatten message under 'data'
			row.update(flatten(msg_dict, prefix='data'))

			# Create writer on first row per topic
			if topic not in writers:
				writers[topic] = ensure_topic_writer(out_dir, topic, row)
			writers[topic].writer.writerow(row)
			count_by_topic[topic] = count_by_topic.get(topic, 0) + 1
	finally:
		for tw in writers.values():
			try:
				tw.file.close()
			except Exception:
				pass

	for t, c in count_by_topic.items():
		print(f"Exported {c} messages for {t} -> {sanitize_topic(t)}.csv")


# -------- Raw fallback using sqlite3 + metadata.yaml --------

def raw_export(bag_dir: str, out_dir: str, topics: Optional[List[str]] = None, combined: bool = False) -> None:
	metadata = load_metadata_yaml(bag_dir)
	db_files = find_db3_files(bag_dir, metadata)
	if not db_files:
		raise FileNotFoundError(f"No .db3 files found in {bag_dir}")

	# Build topic meta mapping from SQLite
	# We'll iterate each db3 file and export messages
	os.makedirs(out_dir, exist_ok=True)

	# In combined mode, write one CSV with topic column
	combined_writer: Optional[csv.DictWriter] = None
	combined_file = None
	if combined:
		combined_file = open(os.path.join(out_dir, 'rosbag2_raw_combined.csv'), 'w', newline='')
		combined_writer = csv.DictWriter(combined_file, fieldnames=['topic', 'type', 'serialization_format', 'time_ns', 'time_sec', 'data_base64'])
		combined_writer.writeheader()

	total_counts: Dict[str, int] = {}

	try:
		for db in db_files:
			conn = sqlite3.connect(db)
			try:
				cur = conn.cursor()
				# topics: id, name, type, serialization_format, offered_qos_profiles
				cur.execute("SELECT id, name, type, serialization_format FROM topics")
				topic_rows = cur.fetchall()
				id_to_topic = {row[0]: (row[1], row[2], row[3]) for row in topic_rows}
				selected_names = set(topics) if topics else {name for (_, (name, _, _)) in id_to_topic.items()}
				ids_to_read = [tid for tid, (name, _, _) in id_to_topic.items() if name in selected_names]
				if not ids_to_read:
					continue

				# If not combined, prepare per-topic writers
				topic_writers: Dict[str, Tuple[csv.DictWriter, Any]] = {}
				if not combined:
					for tid in ids_to_read:
						name, type_str, ser = id_to_topic[tid]
						fname = os.path.join(out_dir, f"{sanitize_topic(name)}_raw.csv")
						f = open(fname, 'w', newline='')
						w = csv.DictWriter(f, fieldnames=['time_ns', 'time_sec', 'data_base64', 'type', 'serialization_format'])
						w.writeheader()
						topic_writers[name] = (w, f)

				# messages: topic_id, timestamp, data
				# Use parameterized IN clause
				placeholders = ','.join('?' for _ in ids_to_read)
				cur.execute(f"SELECT topic_id, timestamp, data FROM messages WHERE topic_id IN ({placeholders}) ORDER BY timestamp ASC", ids_to_read)
				for topic_id, t_ns, data in cur:
					name, type_str, ser = id_to_topic[topic_id]
					total_counts[name] = total_counts.get(name, 0) + 1
					row = {
						'time_ns': int(t_ns),
						'time_sec': float(t_ns) / 1e9,
						'data_base64': base64.b64encode(data).decode('ascii'),
						'type': type_str,
						'serialization_format': ser,
					}
					if combined and combined_writer is not None:
						combined_writer.writerow({'topic': name, **row})
					else:
						w, _f = topic_writers[name]
						w.writerow(row)
			finally:
				conn.close()
	finally:
		if combined_file is not None:
			combined_file.close()

	for t, c in total_counts.items():
		suffix = 'raw.csv' if not combined else 'rosbag2_raw_combined.csv'
		print(f"Exported {c} raw messages for {t} -> {suffix}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(description='Export rosbag2 (.db3 + metadata.yaml) to CSV')
	p.add_argument('bag', help='Path to rosbag2 directory or .db3 file')
	p.add_argument('--out', default='bag_csv', help='Output directory')
	p.add_argument('--topics', nargs='*', default=None, help='List of topics to export (default: all)')
	p.add_argument('--raw', action='store_true', help='Force raw export (no ROS message decoding)')
	p.add_argument('--combined', action='store_true', help='Write a single combined CSV (raw mode only)')
	return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
	args = parse_args(argv)
	bag_dir = resolve_bag_uri(args.bag)
	out_dir = os.path.abspath(args.out)

	if args.raw:
		raw_export(bag_dir, out_dir, topics=args.topics, combined=args.combined)
		return 0

	# Try rich export; fall back to raw if dependencies missing
	try:
		rich_export(bag_dir, out_dir, topics=args.topics)
		return 0
	except Exception as e:
		print(f"Rich export unavailable or failed ({e}); falling back to raw export.")
		raw_export(bag_dir, out_dir, topics=args.topics, combined=args.combined)
		return 0


if __name__ == '__main__':
	sys.exit(main())

