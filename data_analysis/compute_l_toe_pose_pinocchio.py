#!/usr/bin/env python3
"""
Compute the pose of a link (default: L_toe) from a URDF and a long-form joint_states CSV using Pinocchio.

CSV expectations (long/tidy format):
    - Columns: topic, time_sec, joint, position [optional: velocity, effort]
    - Each row corresponds to a single joint measurement at a given time.
    - The joint names must match those in the URDF (e.g., L_hip_joint, L_hip2_joint, ...).

Output: A CSV with time and the link pose in world/base frame
    Columns: time_sec, px, py, pz, qw, qx, qy, qz

Usage:
    python3 data_analysis/compute_l_toe_pose_pinocchio.py \
        --urdf pyroki_scripts/leg05/hr.urdf \
        --csv data_analysis/joint_state_0905_rlpolicy.csv \
        --frame L_toe \
        --topic /joint_states \
        --out data_analysis/l_toe_pose_from_joint_states.csv

Notes:
    - This script uses a fixed-base model by default (root fixed to world).
      If you need a floating base, pass --free-base (but then base pose must be known; not handled here).
    - For performance on large CSVs, only the columns [topic,time_sec,joint,position] are read.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Iterable, List, Optional
import pinocchio as pin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt






LEFT_JOINTS: List[str] = [
    "L_hip_joint",
    "L_hip2_joint",
    "L_thigh_joint",
    "L_calf_joint",
    "L_toe_joint",
]

# Known joint position biases from /joint_states. These are additive corrections that
# should be applied to measurements before forward kinematics.
# Units: radians; keys must match URDF/CSV joint names.
# BIASES: Dict[str, float] = {
#     "L_hip_joint": 0.01659,
#     "L_hip2_joint": 0.01456,
#     "L_thigh_joint": -0.0421,
#     "L_calf_joint": 0.02894,
#     "L_toe_joint": -0.00824,
# }
BIASES: Dict[str, float] = {
    "L_hip_joint": 0.0,
    "L_hip2_joint": 0.0,
    "L_thigh_joint": 0.0,
    "L_calf_joint": 0.0,
    "L_toe_joint": 0.0,
}

# Constant pose translation offset to add to computed (px, py, pz)
# Units: meters; order: [x, y, z]
POSE_OFFSET = np.array([0.1371767, -0.0308362, 0.0353833], dtype=float)
# POSE_OFFSET = np.array([0.0, 0.0, 0.0], dtype=float)

def build_model(urdf_path: str, free_base: bool = False) -> pin.Model:
    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    if free_base:
        model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    else:
        model = pin.buildModelFromUrdf(urdf_path)
    return model


def make_q_index_map(model: pin.Model) -> List[tuple[str, int, int]]:
    """Return a list of (joint_name, q_start, nq) in model order, skipping universe.

    Uses model.names for joint names and accumulates q indices.
    """
    mapping: List[tuple[str, int, int]] = []
    idx = 0
    # Index 0 is 'universe'; start from 1.
    for i in range(1, len(model.names)):
        jname = model.names[i]
        nq = int(model.joints[i].nq)
        mapping.append((jname, idx, nq))
        idx += nq
    assert idx == model.nq, f"Indexing mismatch: built {idx} != model.nq {model.nq}"
    return mapping


def dataframe_from_csv(
    csv_path: str,
    topic: str = "/joint_states",
) -> pd.DataFrame:
    usecols = ["topic", "time_sec", "joint", "position"]
    dtypes = {"topic": "category", "joint": "category"}
    df = pd.read_csv(csv_path, usecols=usecols, dtype=dtypes)
    # Filter topic
    df = df[df["topic"] == topic].copy()
    if df.empty:
        raise ValueError(f"No rows found for topic '{topic}' in {csv_path}")
    # Ensure time sorted
    df = df.sort_values(["time_sec", "joint"])  # stable order
    return df


def pivot_positions(df: pd.DataFrame, joints_of_interest: Iterable[str]) -> pd.DataFrame:
    """Pivot long df -> wide with index=time_sec and columns=joint names (positions).

    Only joints present in the dataframe will appear as columns. Missing joints will be NaN.
    We keep the time index sorted.
    """
    wide = df.pivot_table(
        index="time_sec",
        columns="joint",
        values="position",
        aggfunc="last",
    )
    # Restrict to the set we care about (if present); keep others if needed later
    present = [j for j in joints_of_interest if j in wide.columns]
    if not present:
        raise ValueError("None of the requested joints are present in the CSV")
    # Keep all columns but we'll check presence when constructing q; this avoids repeated reindexing
    wide = wide.sort_index()
    return wide


def compute_link_pose_series(
    model: pin.Model,
    wide_positions: pd.DataFrame,
    frame_name: str,
    default_joint_values: Optional[Dict[str, float]] = None,
    require_joints: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Compute link pose for each time sample.

    - wide_positions: index=time_sec, columns=joint names, values=position (rad)
    - default_joint_values: values used when a joint is missing at a timestamp (defaults to 0)
    - require_joints: if provided, rows missing any of these joints are skipped
    """
    data = model.createData()
    frame_id = model.getFrameId(frame_name)
    if frame_id == len(model.frames):  # not found
        raise ValueError(f"Frame '{frame_name}' not found in model frames.")

    q_map = make_q_index_map(model)
    # Precompute the names to accelerate lookups
    col_set = set(map(str, wide_positions.columns))

    rows = []
    # Iterate chronologically
    for t, row in wide_positions.iterrows():
        # Check required joints
        if require_joints:
            missing_req = [j for j in require_joints if j not in col_set or pd.isna(row.get(j, float("nan")))]
            if missing_req:
                continue  # skip this timestamp

        q = pin.neutral(model)
        for jname, q_start, nq in q_map:
            if nq == 0:
                continue
            # Only 1-DoF revolute joints expected here
            val: float
            if jname in col_set:
                v = row.get(jname)
                if pd.isna(v):
                    # fall back to default
                    val = (default_joint_values or {}).get(jname, 0.0)
                else:
                    val = float(v)
                    # Apply additive bias correction to measured values only
                    if jname in BIASES:
                        val += BIASES[jname]
            else:
                val = (default_joint_values or {}).get(jname, 0.0)
            # Set into q
            if nq == 1:
                q[q_start] = val
            else:
                # For completeness in case of unusual joints (not expected in this URDF)
                for k in range(nq):
                    q[q_start + k] = 0.0

        # FK
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        oMf = data.oMf[frame_id]
        # Apply constant translation offset
        p = oMf.translation + POSE_OFFSET
        R = oMf.rotation
        quat = pin.Quaternion(R)
        # Pinocchio/Eigen stores coeffs as [x, y, z, w]
        x, y, z, w = map(float, quat.coeffs())
        rows.append({
            "time_sec": float(t),
            "px": float(p[0]),
            "py": float(p[1]),
            "pz": float(p[2]),
            "qw": w,
            "qx": x,
            "qy": y,
            "qz": z,
        })

    return pd.DataFrame(rows, columns=["time_sec", "px", "py", "pz", "qw", "qx", "qy", "qz"])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compute link pose (L_toe) from joint_states CSV using Pinocchio")
    parser.add_argument("--urdf", required=True, help="Path to URDF (e.g., pyroki_scripts/leg05/hr.urdf)")
    parser.add_argument("--csv", required=True, help="Path to joint_states CSV (long-form)")
    parser.add_argument("--frame", default="L_toe", help="Link/frame name to compute pose for (default: L_toe)")
    parser.add_argument("--topic", default="/joint_states", help="CSV topic to filter (default: /joint_states)")
    parser.add_argument("--out", default=None, help="Output CSV path; defaults next to input CSV")
    parser.add_argument("--free-base", action="store_true", help="Use a free-flyer root (not typical here)")
    parser.add_argument(
        "--require-left",
        action="store_true",
        help="Skip timestamps missing any left-leg joint (L_hip/L_hip2/L_thigh/L_calf/L_toe)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show a matplotlib plot of position (px, py, pz) vs time",
    )
    parser.add_argument(
        "--zero-time",
        action="store_true",
        help="Subtract the initial timestamp so time starts at 0.0 s (also shifts command windows in plots).",
    )
    # Accept either: five separate numbers per flag, or a single comma/space-separated string via --cmd
    parser.add_argument(
        "--cmd-window",
        dest="_cmd_windows_raw",
        nargs=5,
        metavar=("START", "DURATION", "PX", "PY", "PZ"),
        type=str,
        action="append",
        help=(
            "Add a constant position command window overlay using five values: START DURATION PX PY PZ. "
            "Repeat to add multiple windows."
        ),
    )
    parser.add_argument(
        "--cmd",
        dest="_cmd_windows_raw",
        type=str,
        action="append",
        help=(
            "Alternative syntax for a window as a single string: 'START,DURATION,PX,PY,PZ' or 'START DURATION PX PY PZ'. "
            "Repeat to add multiple windows."
        ),
    )
    parser.add_argument(
        "--split-cmd-windows",
        action="store_true",
        help="If set, also create one zoomed plot per command window (x-limits = [start, start+duration]).",
    )

    args = parser.parse_args(argv)

    urdf_path = os.path.abspath(args.urdf)
    csv_path = os.path.abspath(args.csv)

    model = build_model(urdf_path, free_base=args.free_base)
    # Read CSV and pivot
    df = dataframe_from_csv(csv_path, topic=args.topic)
    wide = pivot_positions(df, joints_of_interest=LEFT_JOINTS)

    require = LEFT_JOINTS if args.require_left else None
    pose_df = compute_link_pose_series(
        model=model,
        wide_positions=wide,
        frame_name=args.frame,
        default_joint_values=None,
        require_joints=require,
    )

    if pose_df.empty:
        print("No poses computed (possibly due to missing required joints).")
        return 2

    out_path = args.out
    if out_path is None:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        out_path = os.path.join(os.path.dirname(csv_path), f"{base}_{args.frame}_pose.csv")
    pose_df.to_csv(out_path, index=False)
    print(f"Saved poses to {out_path}  ({len(pose_df)} rows)")

    # Normalize command windows into a list of tuples: (start, duration, px, py, pz)
    cmd_windows: List[tuple[float, float, float, float, float]] = []
    if getattr(args, "_cmd_windows_raw", None):
        for item in args._cmd_windows_raw:
            if isinstance(item, list):
                parts = item
            else:
                # item is a string; split on comma and whitespace
                parts = [p for p in item.replace(",", " ").split() if p]
            if len(parts) != 5:
                raise SystemExit(f"--cmd/--cmd-window expects 5 values, got: {parts}")
            try:
                start, duration, cx, cy, cz = map(float, parts)
            except ValueError as e:
                raise SystemExit(f"Failed to parse window values {parts}: {e}")
            cmd_windows.append((start, duration, cx, cy, cz))

    # Optionally shift time so it starts at 0.0s
    time_offset = 0.0
    if args.zero_time and not pose_df.empty:
        time_offset = float(pose_df["time_sec"].min())
        if time_offset != 0.0:
            pose_df["time_sec"] = pose_df["time_sec"] - time_offset
            print(f"Zeroed time axis by subtracting t0={time_offset:.6f} s")

    if args.plot:
        # Three stacked subplots: px, py, pz vs time
        fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        t = pose_df["time_sec"].values
        series = [("px", "x (m)"), ("py", "y (m)"), ("pz", "z (m)")]
        colors = {"px": "C0", "py": "C1", "pz": "C2"}
        for ax, (col, ylabel) in zip(axes, series):
            ax.plot(t, pose_df[col].values, label=col, color=colors[col])
            # Overlay command windows, if any
            if cmd_windows:
                first_labels_done = getattr(ax, "_cmd_labels_done", False)
                for win in cmd_windows:
                    start, duration, cx, cy, cz = win
                    # Shift windows if time was zeroed
                    start0 = start - time_offset
                    end = start0 + duration
                    ax.axvspan(start, end, color="k", alpha=0.06)
                    cmd_val = {"px": cx, "py": cy, "pz": cz}[col]
                    label = None if first_labels_done else f"cmd {col}"
                    ax.hlines(cmd_val, xmin=start0, xmax=end, colors=colors[col], linestyles="dashed", label=label)
                ax._cmd_labels_done = True
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
        axes[-1].set_xlabel("time (s)")
        plt.tight_layout()
        plt.show()

        # Optional: split view per command window
        if cmd_windows and args.split_cmd_windows:
            for idx, win in enumerate(cmd_windows, start=1):
                start, duration, cx, cy, cz = win
                start0 = start - time_offset
                end = start0 + duration
                figw, axesw = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
                for ax, (col, ylabel) in zip(axesw, series):
                    ax.plot(t, pose_df[col].values, label=col, color=colors[col])
                    ax.set_xlim(start0, end)
                    ax.axvspan(start0, end, color="k", alpha=0.06)
                    cmd_val = {"px": cx, "py": cy, "pz": cz}[col]
                    ax.hlines(cmd_val, xmin=start0, xmax=end, colors=colors[col], linestyles="dashed", label=f"cmd {col}")
                    ax.set_ylabel(ylabel)
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc="best")
                axesw[-1].set_xlabel("time (s)")
                figw.suptitle(f"Window {idx}: [{start0:.2f}, {end:.2f}] s")
                plt.tight_layout()
                plt.show()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
