"""
Analyze joint-level simulation outputs from play_data.npz.

- Reads npz with keys:
	applied_torques: (T*N, D) or (T, N, D)
	joint_vels:      (T*N, D) or (T, N, D)
	angular_velocities: (T, N, D1) [used to infer T & N if needed]
	project_g:          (T, N, D2) [fallback for inferring T & N]

- Reshapes joint arrays to (T, N, D) and plots timeseries for a
  selected robot and joint index.

Usage:
  python analyze_joint_sim_data.py \
	[--npz data_analysis/play_data.npz] \
	[--robot-index 0] [--joint-index 0] \
	[--T <int> --N <int>] [--outdir path] [--show]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _infer_TN(npz_data: dict) -> Optional[Tuple[int, int]]:
	for key in ("angular_velocities", "project_g"):
		if key in npz_data:
			arr = npz_data[key]
			if isinstance(arr, np.ndarray) and arr.ndim == 3:
				T, N, _ = arr.shape
				return T, N
	return None


def _reshape_TND(arr: np.ndarray, T: int, N: int, name: str) -> np.ndarray:
	if arr.ndim == 3:
		t, n, d = arr.shape
		if t != T or n != N:
			raise ValueError(f"{name} has shape {arr.shape} but expected (T={T}, N={N}, D)")
		return arr
	if arr.ndim != 2:
		raise ValueError(f"{name} expected 2D or 3D array, got shape {arr.shape}")
	M, D = arr.shape
	if M != T * N:
		raise ValueError(
			f"{name} first dim {M} does not match T*N={T*N}; cannot reshape to (T,N,D)"
		)
	return arr.reshape(T, N, D)


def _plot_joint_timeseries(
	joint_vels_TND: np.ndarray,
	torques_TND: np.ndarray,
	robot_idx: int,
	joint_idx: int,
	out_path: str,
	show: bool = False,
) -> None:
	T = joint_vels_TND.shape[0]
	t = np.arange(T)

	v = joint_vels_TND[:, robot_idx, joint_idx]
	tau = torques_TND[:, robot_idx, joint_idx]

	fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

	ax = axes[0]
	ax.plot(t, v, label="joint vel")
	ax.set_title(f"Joint velocity (robot={robot_idx}, joint={joint_idx})")
	ax.set_ylabel("rad/s")
	ax.grid(True, alpha=0.3)

	ax = axes[1]
	ax.plot(t, tau, color="#E45756", label="joint torque")
	ax.set_title(f"Joint torque (robot={robot_idx}, joint={joint_idx})")
	ax.set_xlabel("time step")
	ax.set_ylabel("Nm")
	ax.grid(True, alpha=0.3)

	fig.tight_layout()
	fig.savefig(out_path, dpi=150)
	if show:
		plt.show()
	else:
		plt.close(fig)


def main() -> None:
	parser = argparse.ArgumentParser(description="Reshape joint arrays to (T,N,D) and plot selected joint signals")
	default_npz = os.path.join(os.path.dirname(__file__), "play_data.npz")
	parser.add_argument("--npz", default=default_npz, help="Path to NPZ file")
	parser.add_argument("--robot-index", type=int, default=0, help="Robot index (0..N-1)")
	parser.add_argument("--joint-index", type=int, default=0, help="Joint index (0..D-1)")
	parser.add_argument("--T", type=int, default=1501, help="Time steps T (optional; inferred if possible)")
	parser.add_argument("--N", type=int, default=1, help="Robots N (optional; inferred if possible)")
	parser.add_argument("--outdir", default=None, help="Directory to save the figure (default: data_analysis/out)")
	parser.add_argument("--show", action="store_true", help="Show plot interactively")
	args = parser.parse_args()

	if not os.path.isfile(args.npz):
		print(f"NPZ not found: {args.npz}", file=sys.stderr)
		sys.exit(1)

	with np.load(args.npz) as data:
		keys = list(data.keys())
		required = ["applied_torques", "joint_vels"]
		missing = [k for k in required if k not in data]
		if missing:
			print(f"Missing keys in NPZ: {missing}. Available: {keys}", file=sys.stderr)
			sys.exit(1)
		applied_torques = data["applied_torques"]
		joint_vels = data["joint_vels"]
		# For inferring T and N
		ang = data["angular_velocities"] if "angular_velocities" in data else None
		pg = data["project_g"] if "project_g" in data else None

	inferred = _infer_TN({"angular_velocities": ang, "project_g": pg})
	if args.T is not None and args.N is not None:
		T, N = args.T, args.N
	elif inferred is not None:
		T, N = inferred
	else:
		print("Could not infer T and N; please provide --T and --N", file=sys.stderr)
		sys.exit(1)

	# Reshape to (T,N,D)
	joint_vels_TND = _reshape_TND(joint_vels, T, N, name="joint_vels")
	torques_TND = _reshape_TND(applied_torques, T, N, name="applied_torques")

	# Bounds check for indices
	D = joint_vels_TND.shape[2]
	if not (0 <= args.robot_index < N):
		print(f"robot-index {args.robot_index} out of range [0,{N-1}]", file=sys.stderr)
		sys.exit(1)
	if not (0 <= args.joint_index < D):
		print(f"joint-index {args.joint_index} out of range [0,{D-1}]", file=sys.stderr)
		sys.exit(1)

	# Output path
	outdir = args.outdir or os.path.join(os.path.dirname(__file__), "out")
	os.makedirs(outdir, exist_ok=True)
	out_path = os.path.join(
		outdir,
		f"joint_timeseries_robot{args.robot_index}_joint{args.joint_index}.png",
	)

	print(
		f"Shapes: joint_vels {joint_vels_TND.shape}, applied_torques {torques_TND.shape};"
		f" plotting robot={args.robot_index}, joint={args.joint_index}"
	)
	_plot_joint_timeseries(
		joint_vels_TND,
		torques_TND,
		robot_idx=args.robot_index,
		joint_idx=args.joint_index,
		out_path=out_path,
		show=args.show,
	)
	print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
	main()

