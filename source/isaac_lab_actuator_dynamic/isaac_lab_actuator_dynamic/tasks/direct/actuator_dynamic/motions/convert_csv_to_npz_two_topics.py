#!/usr/bin/env python3
"""
Convert CSV(s) of joint data into a trimmed NPZ segment, explicitly handling two topics
that can have different sampling rates (e.g., /joint_states vs /filtered_joint_cmd).

Inputs:
  - Either one CSV with columns: [time_sec, topic, joint, position] and optional [velocity, effort]
    or two CSVs, one for state, one for command, with the same schema.
  - Time range in seconds (relative window over time_sec; by default time_sec is normalized to start at 0)
  - Output NPZ file path

Key features:
  - Robust alignment of two topics with different sampling rates by resampling onto a common grid
  - Choose to align on a fixed FPS grid, or on the state/cmd native timeline
  - Supports a single joint or all joints present in the CSV(s)

Examples:
  # Single CSV containing both topics
  python3 convert_csv_to_npz_two_topics.py \
      --csv data_analysis/joint_state_latest_3.csv \
      --t0 23 --t1 55 \
      --out motions/recorded_real_motor_data_2.npz --resample-to-fps --fps 500 --plot

  # Two CSVs (e.g., split logs), align to uniform grid at inferred max FPS
  python3 convert_csv_to_npz_two_topics.py \
      --csv-state state.csv --csv-cmd cmd.csv \
      --t0 2 --t1 7 --out motions/segment_L_hip2.npz --joint-name L_hip2_joint --resample-to-fps

Notes:
  - Default topics: --state-topic /joint_states, --cmd-topic /filtered_joint_cmd.
  - When using two CSVs, topic names are only used for plotting labels and can be left as defaults.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt  # optional
except Exception:  # pragma: no cover
    plt = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trim CSV(s) with 2 topics and save as NPZ (handles different sampling rates)")
    g_in = p.add_argument_group("Inputs")
    g_in.add_argument("--csv", type=str, help="Single CSV containing both topics")
    g_in.add_argument("--csv-state", type=str, help="CSV containing state topic rows")
    g_in.add_argument("--csv-cmd", type=str, help="CSV containing command topic rows")

    p.add_argument("--out", required=True, help="Path to output .npz file")
    p.add_argument("--t0", type=float, required=True, help="Start time in seconds (relative to CSV min time)")
    p.add_argument("--t1", type=float, required=True, help="End time in seconds (relative to CSV min time)")
    p.add_argument("--absolute", action="store_true", help="Interpret t0/t1 as absolute (no normalization)")
    p.add_argument("--joint-name", type=str, default="ALL", help="Joint name to extract; 'ALL' for every joint")

    p.add_argument("--state-topic", type=str, default="/joint_states", help="State topic name (label)")
    p.add_argument("--cmd-topic", type=str, default="/filtered_joint_cmd", help="Command topic name (label)")

    p.add_argument("--fps", type=float, default=None, help="Target FPS for uniform grid when resampling")
    p.add_argument(
        "--resample-to-fps",
        action="store_true",
        help="Resample both topics onto a uniform time grid at target FPS using interpolation",
    )
    p.add_argument(
        "--grid-mode",
        choices=["fixed", "state", "cmd"],
        default="fixed",
        help="Time grid to use: fixed uniform FPS, or the native timeline of state/cmd",
    )
    p.add_argument("--interp", choices=["linear", "nearest"], default="linear", help="Interpolation method")
    p.add_argument("--plot", action="store_true", help="Show quick plots of the trimmed segment")
    p.add_argument("--max-plots", type=int, default=12, help="Max joints to plot when ALL")
    p.add_argument("--plot-cols", type=int, default=3, help="Columns for ALL-joints plot grid")
    return p.parse_args()


def _fill_nan_1d(y: np.ndarray, ts: np.ndarray | None = None) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return np.zeros_like(y)
    m = np.isfinite(y)
    if not np.any(m):
        return np.zeros_like(y)
    out = y.copy()
    x = np.arange(y.size, dtype=float) if ts is None else np.asarray(ts, dtype=float)
    x = x[: y.size]
    if np.any(~m):
        x_valid = x[m]
        y_valid = out[m]
        out[~m] = np.interp(x[~m], x_valid, y_valid)
        first_idx = np.argmax(m)
        last_idx = len(m) - 1 - np.argmax(m[::-1])
        if first_idx > 0:
            out[:first_idx] = out[first_idx]
        if last_idx < y.size - 1:
            out[last_idx + 1 :] = out[last_idx]
    return out


def _fill_nan_2d(Y: np.ndarray, ts: np.ndarray | None = None) -> np.ndarray:
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        return Y
    out = np.empty_like(Y, dtype=float)
    for j in range(Y.shape[1]):
        out[:, j] = _fill_nan_1d(Y[:, j], ts)
    return out


def _estimate_velocity_from_pos(pos: np.ndarray, ts: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos, dtype=float)
    ts = np.asarray(ts, dtype=float)
    if pos.size == 0:
        return np.zeros_like(pos)
    if pos.ndim == 1:
        v = np.gradient(pos, ts)
        return _fill_nan_1d(v, ts)
    v = np.empty_like(pos, dtype=float)
    for j in range(pos.shape[1]):
        v[:, j] = _fill_nan_1d(np.gradient(pos[:, j], ts), ts)
    return v


def infer_fps(ts: np.ndarray) -> float:
    if ts is None or ts.size < 3:
        return 200.0
    dt = np.diff(ts)
    dt = dt[dt > 0]
    if dt.size == 0:
        return 200.0
    med = float(np.median(dt))
    return 200.0 if med <= 0 else float(1.0 / med)


def make_time_grid(t0: float, t1: float, fps: float) -> np.ndarray:
    if fps <= 0:
        raise ValueError("fps must be positive")
    step = 1.0 / float(fps)
    n = int(np.floor(max(0.0, (t1 - t0)) * float(fps))) + 1
    ts = t0 + np.arange(n, dtype=float) * step
    if ts.size and ts[-1] > t1 + 1e-9:
        ts = ts[:-1]
    return ts


def interp_1d(ts_src: np.ndarray, y_src: np.ndarray, ts_dst: np.ndarray, method: str = "linear") -> np.ndarray:
    if ts_src is None or y_src is None:
        return np.full_like(ts_dst, np.nan, dtype=float)
    ts_src = np.asarray(ts_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    ts_dst = np.asarray(ts_dst, dtype=float)
    m = np.isfinite(ts_src) & np.isfinite(y_src)
    ts = ts_src[m]
    ys = y_src[m]
    if ts.size == 0:
        return np.full_like(ts_dst, np.nan, dtype=float)
    order = np.argsort(ts)
    ts = ts[order]
    ys = ys[order]
    if method == "nearest":
        idx = np.searchsorted(ts, ts_dst, side="left")
        idx_clip = np.clip(idx, 0, ts.size - 1)
        left = np.clip(idx - 1, 0, ts.size - 1)
        right = idx_clip
        choose_right = (ts_dst - ts[left]) > (ts[right] - ts_dst)
        nearest_idx = np.where(choose_right, right, left)
        out = ys[nearest_idx].astype(float, copy=False)
    else:
        out = np.interp(ts_dst, ts, ys).astype(float, copy=False)
    oob = (ts_dst < ts[0]) | (ts_dst > ts[-1])
    if np.any(oob):
        out = out.copy()
        out[oob] = np.nan
    return out


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"time_sec", "topic", "joint", "position"}
    if not req.issubset(df.columns):
        raise ValueError(f"CSV {path} must contain columns {sorted(req)}; got {sorted(df.columns)}")
    return df


def _normalize_time(df: pd.DataFrame, absolute: bool) -> pd.DataFrame:
    if absolute:
        return df
    t0 = float(df["time_sec"].min())
    if np.isfinite(t0):
        df = df.copy()
        df["time_sec"] = df["time_sec"] - t0
    return df


def _slice_time(df: pd.DataFrame, t0: float, t1: float) -> pd.DataFrame:
    return df[(df["time_sec"] >= t0) & (df["time_sec"] <= t1)].copy()


def main() -> None:
    args = parse_args()

    # Load CSV(s)
    if args.csv_state or args.csv_cmd:
        if not (args.csv_state and args.csv_cmd):
            raise ValueError("When using --csv-state/--csv-cmd, provide both")
        df_state = _read_csv(Path(args.csv_state))
        df_cmd = _read_csv(Path(args.csv_cmd))
        # Normalize independently then later align by window
        df_state = _normalize_time(df_state, args.absolute)
        df_cmd = _normalize_time(df_cmd, args.absolute)
    elif args.csv:
        df = _read_csv(Path(args.csv))
        df = _normalize_time(df, args.absolute)
        df_state = df[df["topic"] == args.state_topic].copy()
        df_cmd = df[df["topic"] == args.cmd_topic].copy()
    else:
        raise ValueError("Provide either --csv or both --csv-state and --csv-cmd")

    # Validate window
    t0, t1 = float(args.t0), float(args.t1)
    if t1 <= t0:
        raise ValueError("t1 must be greater than t0")

    # Slice to window first (keeps memory lower)
    st_win = _slice_time(df_state, t0, t1)
    cm_win = _slice_time(df_cmd, t0, t1)

    # Sort
    st_win.sort_values(["time_sec", "joint"], inplace=True)
    cm_win.sort_values(["time_sec", "joint"], inplace=True)

    joint = args.joint_name
    all_joints = isinstance(joint, str) and joint.strip().upper() in {"ALL", "ALL_JOINTS"}

    if all_joints:
        # Wide matrices on their OWN timelines first
        pos_st_w = st_win.pivot_table(index="time_sec", columns="joint", values="position", aggfunc="first").sort_index()
        vel_st_w = (
            st_win.pivot_table(index="time_sec", columns="joint", values="velocity", aggfunc="first").sort_index()
            if "velocity" in st_win.columns
            else None
        )
        eff_st_w = (
            st_win.pivot_table(index="time_sec", columns="joint", values="effort", aggfunc="first").sort_index()
            if "effort" in st_win.columns and st_win["effort"].notna().any()
            else None
        )
        cmd_w = cm_win.pivot_table(index="time_sec", columns="joint", values="position", aggfunc="first").sort_index()

        # Decide grid
        ts_st = pos_st_w.index.to_numpy(float)
        ts_cm = cmd_w.index.to_numpy(float)
        fps_state = infer_fps(ts_st)
        fps_cmd = infer_fps(ts_cm)
        fps = float(args.fps) if args.fps is not None else max(fps_state, fps_cmd)

        if args.grid_mode == "state":
            ts_grid = ts_st
        elif args.grid_mode == "cmd":
            ts_grid = ts_cm
        elif args.resample_to_fps:
            ts_grid = make_time_grid(t0, t1, fps)
        else:
            # default to state timeline if not resampling
            ts_grid = ts_st

        # Resample helper
        def resample_wide(arr_w: Optional[pd.DataFrame]) -> Optional[np.ndarray]:
            if arr_w is None:
                return None
            times = arr_w.index.to_numpy(dtype=float)
            out = np.empty((ts_grid.size, arr_w.shape[1]), dtype=float)
            out[:] = np.nan
            for j, col in enumerate(arr_w.columns):
                out[:, j] = interp_1d(times, arr_w[col].to_numpy(dtype=float), ts_grid, method=args.interp)
            return out

        dof_positions = resample_wide(pos_st_w)
        dof_velocities = resample_wide(vel_st_w) if vel_st_w is not None else None
        dof_efforts = resample_wide(eff_st_w) if eff_st_w is not None else None
        dof_position_commands = resample_wide(cmd_w)
        time_sec = ts_grid

        # Fill/estimate
        dof_positions = _fill_nan_2d(dof_positions, time_sec)
        if dof_velocities is None or not np.isfinite(dof_velocities).any():
            dof_velocities = _estimate_velocity_from_pos(dof_positions, time_sec)
        else:
            dof_velocities = _fill_nan_2d(dof_velocities, time_sec)
        if dof_efforts is None or not np.isfinite(dof_efforts).any():
            dof_efforts = np.zeros_like(dof_positions)
        else:
            dof_efforts = _fill_nan_2d(dof_efforts, time_sec)
        dof_position_commands = _fill_nan_2d(dof_position_commands, time_sec)
        nan_cols = ~np.isfinite(dof_position_commands).any(axis=0)
        if np.any(nan_cols):
            dof_position_commands[:, nan_cols] = dof_positions[:, nan_cols]

        dof_names = pos_st_w.columns.to_numpy(object)
        data = {
            "fps": np.array(fps),
            "time_sec": np.asarray(time_sec, dtype=float),
            "dof_names": dof_names,
            "dof_positions": dof_positions,
            "dof_velocities": dof_velocities,
            "dof_efforts": dof_efforts,
            "dof_position_commands": dof_position_commands,
        }
    else:
        # Single joint path
        st_j = st_win[st_win["joint"] == joint].copy()
        cm_j = cm_win[cm_win["joint"] == joint].copy()
        st_j.sort_values("time_sec", inplace=True)
        cm_j.sort_values("time_sec", inplace=True)

        ts_st = st_j["time_sec"].to_numpy(float)
        ts_cm = cm_j["time_sec"].to_numpy(float)
        pos = st_j["position"].to_numpy(float)
        vel = st_j["velocity"].to_numpy(float) if "velocity" in st_j.columns else np.array([])
        if "effort" in st_j.columns and st_j["effort"].notna().any():
            eff = st_j["effort"].to_numpy(float)
            ts_eff = ts_st
        elif "effort" in cm_j.columns and cm_j["effort"].notna().any():
            eff = cm_j["effort"].to_numpy(float)
            ts_eff = ts_cm
        else:
            eff = np.array([])
            ts_eff = ts_st
        cmd = cm_j["position"].to_numpy(float)

        fps_state = infer_fps(ts_st)
        fps_cmd = infer_fps(ts_cm)
        fps = float(args.fps) if args.fps is not None else max(fps_state, fps_cmd)

        if args.grid_mode == "state":
            ts_grid = ts_st
        elif args.grid_mode == "cmd":
            ts_grid = ts_cm
        elif args.resample_to_fps:
            ts_grid = make_time_grid(t0, t1, fps)
        else:
            ts_grid = ts_st

        pos_rs = interp_1d(ts_st, pos, ts_grid, method=args.interp) if ts_st.size else np.full(ts_grid.size, np.nan)
        vel_rs = (
            interp_1d(ts_st, vel, ts_grid, method=args.interp) if isinstance(vel, np.ndarray) and vel.size else np.full(ts_grid.size, np.nan)
        )
        eff_rs = (
            interp_1d(ts_eff, eff, ts_grid, method=args.interp) if isinstance(eff, np.ndarray) and eff.size else np.full(ts_grid.size, np.nan)
        )
        cmd_rs = interp_1d(ts_cm, cmd, ts_grid, method=args.interp) if ts_cm.size else np.full(ts_grid.size, np.nan)

        pos_rs = _fill_nan_1d(pos_rs, ts_grid)
        if not np.isfinite(vel_rs).any():
            vel_rs = _estimate_velocity_from_pos(pos_rs, ts_grid)
        else:
            vel_rs = _fill_nan_1d(vel_rs, ts_grid)
        if not np.isfinite(eff_rs).any():
            eff_rs = np.zeros_like(pos_rs)
        else:
            eff_rs = _fill_nan_1d(eff_rs, ts_grid)
        cmd_rs = _fill_nan_1d(cmd_rs, ts_grid)
        if not np.isfinite(cmd_rs).any():
            cmd_rs = pos_rs.copy()

        data = {
            "fps": np.array(fps),
            "time_sec": ts_grid,
            "dof_names": np.array([joint]),
            "dof_positions": pos_rs,
            "dof_velocities": vel_rs,
            "dof_efforts": eff_rs,
            "dof_position_commands": cmd_rs,
        }

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data)

    # Plot
    if args.plot and plt is not None:
        import matplotlib.pyplot as _plt
        if all_joints:
            pos = np.asarray(data["dof_positions"], dtype=float)
            cmd = np.asarray(data["dof_position_commands"], dtype=float)
            names = np.asarray(data["dof_names"], dtype=object)
            ts = np.asarray(data["time_sec"], dtype=float)
            if pos.ndim == 2 and cmd.ndim == 2:
                n_j = names.size
                n_plot = min(int(args.max_plots), n_j)
                cols = max(1, int(args.plot_cols))
                rows = (n_plot + cols - 1) // cols
                _plt.figure(figsize=(4.0 * cols + 2, 2.5 * rows + 1))
                for i in range(n_plot):
                    _plt.subplot(rows, cols, i + 1)
                    _plt.plot(ts, pos[:, i], label="state pos")
                    _plt.plot(ts, cmd[:, i], label="cmd pos", linestyle="--")
                    _plt.title(str(names[i]))
                    _plt.xlabel("time (s)")
                    _plt.ylabel("pos (rad)")
                    _plt.legend(fontsize=8)
                _plt.tight_layout(); _plt.show()
        else:
            ts = np.asarray(data["time_sec"], dtype=float)
            _plt.figure(figsize=(12, 8))
            _plt.subplot(4, 1, 1); _plt.plot(ts, data["dof_positions"], label="state pos"); _plt.legend(); _plt.title("Position")
            _plt.subplot(4, 1, 2); _plt.plot(ts, data["dof_velocities"], label="state vel"); _plt.legend(); _plt.title("Velocity")
            _plt.subplot(4, 1, 3); _plt.plot(ts, data["dof_efforts"], label="effort"); _plt.legend(); _plt.title("Effort")
            _plt.subplot(4, 1, 4); _plt.plot(ts, data["dof_position_commands"], label="cmd pos"); _plt.legend(); _plt.title("Command")
            _plt.tight_layout(); _plt.show()

    # Summary
    pos_shape = np.shape(data.get("dof_positions"))
    n_joints = len(data.get("dof_names", []))
    print(f"Wrote {out_path} | range=({t0:.2f}, {t1:.2f}) | fps={float(data['fps']):.2f} | joints={n_joints} | pos_shape={pos_shape}")


if __name__ == "__main__":
    main()
