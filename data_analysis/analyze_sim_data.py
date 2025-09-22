"""
Analyze simulation NPZ output and plot histograms per dimension.

Expect NPZ created via np.savez with keys:
  - applied_torques: any shape (unused here)
  - angular_velocities: (T, N, D)
  - joint_vels: any shape (unused here)
  - project_g: (T, N, D)

We flatten (T, N, D) -> (T*N, D) and plot subplots of histograms
for each dimension for both angular_velocities and project_g.

Usage:
  python analyze_sim_data.py [--npz path] [--bins 80] [--show] [--outdir path]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


def _flatten_txn_d(arr: np.ndarray, name: str) -> np.ndarray:
	if arr.ndim != 3:
		raise ValueError(f"{name} expected 3D array (T,N,D), got shape {arr.shape}")
	T, N, D = arr.shape
	return arr.reshape(T * N, D)


def _plot_dim_histograms(data: np.ndarray, title: str, bins: int = 80) -> Tuple[plt.Figure, np.ndarray]:
	"""Plot per-dimension histograms for 2D data of shape (M, D)."""
	if data.ndim != 2:
		raise ValueError(f"Histogram input must be 2D (M,D), got {data.shape}")
	M, D = data.shape
	# Layout: 1 x D row (common for D<=6). If larger, make grid.
	ncols = min(D, 4)
	nrows = (D + ncols - 1) // ncols
	fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols + 2, 3.5 * nrows))
	axes = np.atleast_1d(axes).ravel()

	for d in range(D):
		ax = axes[d]
		ax.hist(
			data[:, d],
			bins=bins,
			color="#4C78A8",
			alpha=0.85,
			edgecolor="white",
			density=True,  # normalize to probability density
		)
		ax.set_title(f"{title}[{d}]")
		ax.set_xlabel("value")
		ax.set_ylabel("density")
		ax.grid(True, alpha=0.25)

	# Hide any unused axes
	for k in range(D, len(axes)):
		axes[k].axis("off")

	fig.suptitle(f"Histograms per dimension: {title} (M={M}, D={D})")
	fig.tight_layout(rect=(0, 0, 1, 0.95))
	return fig, axes


def main() -> None:
	parser = argparse.ArgumentParser(description="Flatten (T,N,D) arrays and plot per-dimension histograms")
	default_npz = os.path.join(os.path.dirname(__file__), "play_data_2.npz")
	parser.add_argument("--npz", default=default_npz, help="Path to NPZ file (default: data_analysis/play_data.npz)")
	parser.add_argument("--bins", type=int, default=80, help="Histogram bins")
	parser.add_argument("--show", action="store_true", help="Show figures interactively")
	parser.add_argument("--outdir", default=None, help="Directory to save figures; defaults to data_analysis/out")
	parser.add_argument(
		"--real-csv",
		default=None,
		help="Optional: path to real IMU CSV (decoded). If provided, also plot histograms for real data",
	)
	args = parser.parse_args()

	if not os.path.isfile(args.npz):
		print(f"NPZ not found: {args.npz}", file=sys.stderr)
		sys.exit(1)

	with np.load(args.npz) as data:
		missing = [k for k in ("angular_velocities", "project_g") if k not in data]
		if missing:
			print(f"Missing keys in NPZ: {missing}", file=sys.stderr)
			print(f"Available keys: {list(data.keys())}", file=sys.stderr)
			sys.exit(1)

		ang = data["angular_velocities"]
		pg = data["project_g"]

	print(f"angular_velocities shape: {ang.shape}")
	print(f"project_g shape:         {pg.shape}")

	# Flatten (T,N,D) -> (T*N, D)
	# ang2 = _flatten_txn_d(ang, "angular_velocities")
	# pg2 = _flatten_txn_d(pg, "project_g")
	# print(f"Flattened: angular_velocities -> {ang2.shape}")
	# print(f"Flattened: project_g         -> {pg2.shape}")
	ang2 = ang
	pg2 = pg

	# Output directory
	outdir = args.outdir or os.path.join(os.path.dirname(__file__), "out")
	os.makedirs(outdir, exist_ok=True)

	# Plot and save
	fig1, _ = _plot_dim_histograms(ang2, title="angular_velocities", bins=args.bins)
	f1_path = os.path.join(outdir, "hist_angular_velocities.png")
	fig1.savefig(f1_path, dpi=150)
	print(f"Saved: {f1_path}")

	fig2, _ = _plot_dim_histograms(pg2, title="project_g", bins=args.bins)
	f2_path = os.path.join(outdir, "hist_project_g.png")
	fig2.savefig(f2_path, dpi=150)
	print(f"Saved: {f2_path}")

	# Optional: real IMU data histograms (wx,wy,wz) and projected g (gbx,gby,gbz)
	if args.real_csv is not None:
		try:
			from analyze_imu_data import load_imu_quat_angrate, compute_g_in_base
			import pandas as pd  # noqa: F401
		except Exception as e:
			print(f"Failed to import real data utilities: {e}", file=sys.stderr)
		else:
			df = load_imu_quat_angrate(args.real_csv)
			df = compute_g_in_base(df)
			real_ang = df[["wx", "wy", "wz"]].to_numpy(dtype=float)
			real_g = df[["gbx", "gby", "gbz"]].to_numpy(dtype=float)

			fig3, _ = _plot_dim_histograms(real_ang, title="real_angular_rates", bins=args.bins)
			f3_path = os.path.join(outdir, "hist_real_angular_rates.png")
			fig3.savefig(f3_path, dpi=150)
			print(f"Saved: {f3_path}")

			fig4, _ = _plot_dim_histograms(real_g, title="real_project_g", bins=args.bins)
			f4_path = os.path.join(outdir, "hist_real_project_g.png")
			fig4.savefig(f4_path, dpi=150)
			print(f"Saved: {f4_path}")

			if not args.show:
				plt.close(fig3)
				plt.close(fig4)

	if args.show:
		plt.show()
	else:
		plt.close(fig1)
		plt.close(fig2)


if __name__ == "__main__":
	main()

