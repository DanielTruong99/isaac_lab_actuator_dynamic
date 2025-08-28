#!/usr/bin/env python3
"""
Quick viewer for NPZ segments produced by convert_csv_to_npz.py.

Usage examples:
  python3 plot_npz_segment.py --npz recorded_real_motor_data_2.npz
  python3 plot_npz_segment.py --npz recorded_real_motor_data_2.npz --max-plots 8 --plot-cols 4
  python3 plot_npz_segment.py --npz recorded_real_motor_data_2.npz --joint-name L_hip2

Notes:
- Handles both ALL-joint (2D arrays) and single-joint (1D arrays) outputs.
- If matplotlib isn't available, prints a summary and exits.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None  # Plotting optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot NPZ segment exported by convert_csv_to_npz.py")
    p.add_argument("--npz", required=True, help="Path to .npz file")
    p.add_argument("--joint-name", type=str, default=None, help="Joint name to plot (if ALL). If omitted, grid plot")
    p.add_argument("--joint-index", type=int, default=None, help="Joint index to plot (0-based) when --joint-name not given")
    p.add_argument("--max-plots", type=int, default=12, help="Max joints to plot in grid mode")
    p.add_argument("--plot-cols", type=int, default=3, help="Number of columns for grid mode")
    return p.parse_args()


def _has_non_nan(arr: Optional[np.ndarray]) -> bool:
    """Return True if the array exists, has elements, and not all values are NaN."""
    if arr is None:
        return False
    a = np.asarray(arr)
    if a.size == 0:
        return False
    try:
        return np.any(~np.isnan(a))
    except Exception:
        # Non-float types; assume valid if any element exists
        return a.size > 0


def _get_col(arr: Optional[np.ndarray], idx: int) -> Optional[np.ndarray]:
    """Safely get a column from a (T, J) array; returns None if arr is None."""
    if arr is None:
        return None
    return np.asarray(arr[:, idx], dtype=float)


def main() -> None:
    args = parse_args()
    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    # Extract common arrays if present
    time_sec = data.get("time_sec")
    dof_names = data.get("dof_names")
    dof_positions = data.get("dof_positions")
    dof_velocities = data.get("dof_velocities")
    dof_efforts = data.get("dof_efforts")
    dof_position_commands = data.get("dof_position_commands")
    fps = data.get("fps")

    # Normalize dtypes
    if dof_names is not None:
        try:
            dof_names = np.asarray(dof_names, dtype=object)
        except Exception:
            pass
    if time_sec is not None:
        time_sec = np.asarray(time_sec, dtype=float)

    # Print summary
    print("== NPZ Summary ==")
    print(f"file: {npz_path}")
    print(f"fps: {float(fps) if fps is not None else 'n/a'}")
    print(f"time_sec.shape: {None if time_sec is None else time_sec.shape}")
    print(f"dof_names: {None if dof_names is None else list(dof_names)}")
    def shp(x):
        return None if x is None else np.shape(x)
    print(f"positions: {shp(dof_positions)} | velocities: {shp(dof_velocities)} | efforts: {shp(dof_efforts)} | cmds: {shp(dof_position_commands)}")

    if plt is None:
        print("matplotlib not available; install it to view plots.")
        return

    # Determine plotting mode
    # ALL-joint mode if positions are 2D (T, J)
    if dof_positions is not None and np.ndim(dof_positions) == 2:
        T, J = dof_positions.shape
        names = dof_names if dof_names is not None else np.array([f"joint_{i}" for i in range(J)], dtype=object)
        ts = time_sec if time_sec is not None else np.arange(T, dtype=float)

        # Print NaN summary for arrays (per joint)
        def _nan_cols(arr: Optional[np.ndarray]):
            if arr is None:
                return None, None
            try:
                a = np.asarray(arr, dtype=float)
                any_nan = np.any(np.isnan(a), axis=0)
                all_nan = np.all(np.isnan(a), axis=0)
                return any_nan, all_nan
            except Exception:
                return None, None

        for label, arr in (
            ("positions", dof_positions),
            ("velocities", dof_velocities),
            ("efforts", dof_efforts),
            ("cmds", dof_position_commands),
        ):
            any_nan, all_nan = _nan_cols(arr)
            if any_nan is None:
                continue
            any_list = [str(names[i]) for i in range(J) if any_nan[i]]
            all_list = [str(names[i]) for i in range(J) if all_nan[i]]
            if any_list:
                print(f"NaNs detected in {label}: any-nan joints = {any_list}")
            if all_list:
                print(f"All-NaN in {label}: joints = {all_list}")

        if args.joint_name is not None or args.joint_index is not None:
            # Single selected joint from ALL
            if args.joint_name is not None and dof_names is not None:
                try:
                    idx = int(np.where(names == args.joint_name)[0][0])
                except Exception:
                    raise ValueError(f"Joint name '{args.joint_name}' not found in {list(names)}")
            else:
                idx = int(args.joint_index or 0)
                if not (0 <= idx < J):
                    raise IndexError(f"joint-index {idx} out of range [0, {J-1}]")

            pos = _get_col(dof_positions, idx)
            vel = _get_col(dof_velocities, idx)
            eff = _get_col(dof_efforts, idx)
            cmd = _get_col(dof_position_commands, idx)
            name = str(names[idx])

            # Print NaN status for selected joint
            def _count_nan(a: Optional[np.ndarray]) -> int:
                if a is None:
                    return 0
                try:
                    return int(np.isnan(a).sum())
                except Exception:
                    return 0
            for label, a in (("pos", pos), ("vel", vel), ("eff", eff), ("cmd", cmd)):
                if a is None:
                    continue
                cnt = _count_nan(a)
                if cnt:
                    print(f"[{name}] {label}: {cnt} NaN samples (all-NaN={np.all(np.isnan(a))})")

            plt.figure(figsize=(12, 8))
            # Position vs cmd
            ax = plt.subplot(4, 1, 1)
            if _has_non_nan(pos):
                ax.plot(ts[: pos.size], pos, label="state pos")
            else:
                ax.text(0.02, 0.8, "position: all NaN/empty", transform=ax.transAxes, color="red")
            if _has_non_nan(cmd):
                ax.plot(ts[: cmd.size], cmd, label="cmd pos", linestyle="--")
            ax.set_title(f"{name} - Position")
            ax.set_ylabel("pos (rad)")
            ax.legend()
            # Velocity
            ax = plt.subplot(4, 1, 2)
            if _has_non_nan(vel):
                ax.plot(ts[: vel.size], vel, color="orange", label="state vel")
            else:
                ax.text(0.02, 0.8, "velocity: all NaN/empty", transform=ax.transAxes, color="red")
            ax.set_title("Velocity")
            ax.set_ylabel("vel (rad/s)")
            ax.legend()
            # Effort
            ax = plt.subplot(4, 1, 3)
            if _has_non_nan(eff):
                ax.plot(ts[: eff.size], eff, color="green", label="effort")
            else:
                ax.text(0.02, 0.8, "effort: all NaN/empty", transform=ax.transAxes, color="red")
            ax.set_title("Effort")
            ax.set_ylabel("effort (Nm)")
            ax.legend()
            # Command only
            ax = plt.subplot(4, 1, 4)
            if _has_non_nan(cmd):
                ax.plot(ts[: cmd.size], cmd, color="red", label="cmd pos")
            else:
                ax.text(0.02, 0.8, "cmd: all NaN/empty", transform=ax.transAxes, color="red")
            ax.set_title("Position Command")
            ax.set_ylabel("cmd (rad)")
            ax.legend()
            plt.tight_layout()
            plt.show()
            return

        # Grid mode: plot up to max_plots of pos vs cmd
        n_plot = min(int(args.max_plots), J)
        cols = max(1, int(args.plot_cols))
        rows = (n_plot + cols - 1) // cols
        plt.figure(figsize=(4.0 * cols + 2, 2.6 * rows + 1))
        for i in range(n_plot):
            ax = plt.subplot(rows, cols, i + 1)
            y_pos = _get_col(dof_positions, i)
            if _has_non_nan(y_pos):
                ax.plot(ts[: y_pos.size], y_pos, label="state pos")
            else:
                ax.text(0.05, 0.7, "pos: all NaN", transform=ax.transAxes, color="red", fontsize=8)
            y_cmd = _get_col(dof_position_commands, i)
            if _has_non_nan(y_cmd):
                ax.plot(ts[: y_cmd.size], y_cmd, linestyle="--", label="cmd pos")
            ax.set_title(str(names[i]))
            ax.set_xlabel("time (s)")
            ax.set_ylabel("pos (rad)")
            ax.legend(fontsize=8)
        plt.suptitle(f"{npz_path.name} — {J} joints (showing {n_plot})", y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()
        return

    # Single joint arrays (1D)
    if dof_positions is not None and np.ndim(dof_positions) == 1:
        ts = time_sec if time_sec is not None else np.arange(np.size(dof_positions), dtype=float)
        name = str(dof_names[0]) if (dof_names is not None and len(dof_names)) else "joint"
        pos = np.asarray(dof_positions, dtype=float)
        vel = np.asarray(dof_velocities, dtype=float) if dof_velocities is not None else None
        eff = np.asarray(dof_efforts, dtype=float) if dof_efforts is not None else None
        cmd = np.asarray(dof_position_commands, dtype=float) if dof_position_commands is not None else None

        # Print NaN status for 1D arrays
        def _print_1d(label: str, a: Optional[np.ndarray]):
            if a is None:
                return
            try:
                a = np.asarray(a, dtype=float)
                cnt = int(np.isnan(a).sum())
                if cnt:
                    print(f"[{name}] {label}: {cnt} NaN samples (all-NaN={np.all(np.isnan(a))})")
            except Exception:
                pass
        _print_1d("pos", pos)
        _print_1d("vel", vel)
        _print_1d("eff", eff)
        _print_1d("cmd", cmd)

        plt.figure(figsize=(12, 8))
        ax = plt.subplot(4, 1, 1)
        if _has_non_nan(pos):
            ax.plot(ts[: pos.size], pos, label="state pos")
        else:
            ax.text(0.02, 0.8, "position: all NaN/empty", transform=ax.transAxes, color="red")
        if _has_non_nan(cmd):
            ax.plot(ts[: cmd.size], cmd, label="cmd pos", linestyle="--")
        ax.set_title(f"{name} - Position")
        ax.set_ylabel("pos (rad)")
        ax.legend()

        ax = plt.subplot(4, 1, 2)
        if _has_non_nan(vel):
            ax.plot(ts[: vel.size], vel, color="orange", label="state vel")
        else:
            ax.text(0.02, 0.8, "velocity: all NaN/empty", transform=ax.transAxes, color="red")
        ax.set_title("Velocity")
        ax.set_ylabel("vel (rad/s)")
        ax.legend()

        ax = plt.subplot(4, 1, 3)
        if _has_non_nan(eff):
            ax.plot(ts[: eff.size], eff, color="green", label="effort")
        else:
            ax.text(0.02, 0.8, "effort: all NaN/empty", transform=ax.transAxes, color="red")
        ax.set_title("Effort")
        ax.set_ylabel("effort (Nm)")
        ax.legend()

        ax = plt.subplot(4, 1, 4)
        if _has_non_nan(cmd):
            ax.plot(ts[: cmd.size], cmd, color="red", label="cmd pos")
        else:
            ax.text(0.02, 0.8, "cmd: all NaN/empty", transform=ax.transAxes, color="red")
        ax.set_title("Position Command")
        ax.set_ylabel("cmd (rad)")
        ax.legend()

        plt.tight_layout()
        plt.show()
        return

    print("Nothing to plot: unexpected array shapes in NPZ.")


if __name__ == "__main__":
    main()
