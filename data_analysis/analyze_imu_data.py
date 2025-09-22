"""
IMU data reader for VectorNav common group decoded CSV.

- Reads quaternion: data.quaternion.{x,y,z,w}
- Reads angular rate: data.angularrate.{x,y,z}
- Also pulls a timestamp column (bag_time_sec if present, else stamp_sec+stamp_nanosec).

Usage:
  python analyze_imu_data.py [--csv path]

Prints basic info and the first few rows.
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


QUAT_COLS: List[str] = [
    "data.quaternion.x",
    "data.quaternion.y",
    "data.quaternion.z",
    "data.quaternion.w",
]

ANG_RATE_COLS: List[str] = [
    "data.angularrate.x",
    "data.angularrate.y",
    "data.angularrate.z",
]

TIME_PREF_COLS: List[str] = [
    "bag_time_sec",  # float seconds
    # fallback pieces to synthesize time if needed
    "stamp_sec",
    "stamp_nanosec",
]


def load_imu_quat_angrate(csv_path: str) -> pd.DataFrame:
    """Load quaternion and angular rate (gyro) from decoded CSV.

    Returns a DataFrame with columns:
      - time (float seconds)
      - qx, qy, qz, qw
      - wx, wy, wz
    """
    usecols = [
        *QUAT_COLS,
        *ANG_RATE_COLS,
        *TIME_PREF_COLS,
    ]

    # Read only needed columns; ignore extras
    header_cols = pd.read_csv(csv_path, nrows=0).columns
    df = pd.read_csv(csv_path, usecols=[c for c in usecols if c in header_cols])

    cols = df.columns
    # Determine time column
    if "bag_time_sec" in cols:
        time = df["bag_time_sec"].astype(float)
    elif {"stamp_sec", "stamp_nanosec"}.issubset(cols):
        time = df["stamp_sec"].astype(float) + df["stamp_nanosec"].astype(float) * 1e-9
    else:
        raise ValueError("No suitable time columns found (bag_time_sec or stamp_sec+stamp_nanosec)")

    out = pd.DataFrame(
        {
            "time": time,
            "qx": df[QUAT_COLS[0]].astype(float),
            "qy": df[QUAT_COLS[1]].astype(float),
            "qz": df[QUAT_COLS[2]].astype(float),
            "qw": df[QUAT_COLS[3]].astype(float),
            "wx": df[ANG_RATE_COLS[0]].astype(float),
            "wy": df[ANG_RATE_COLS[1]].astype(float),
            "wz": df[ANG_RATE_COLS[2]].astype(float),
        }
    )
    return out


def compute_g_in_base(df: pd.DataFrame) -> pd.DataFrame:
        """Project global unit gravity g=[0,0,1] into base frame using quaternion.

        Assumes quaternion (qx,qy,qz,qw) represents world_from_base orientation.
        Then g_base = R_wb^T * g_world. Using quaternion algebra this simplifies to:
            gbx = 2*(qz*qx - qy*qw)
            gby = 2*(qz*qy + qx*qw)
            gbz = 1 - 2*(qx*qx + qy*qy)
        """
        qx = df["qx"].to_numpy()
        qy = df["qy"].to_numpy()
        qz = df["qz"].to_numpy()
        qw = df["qw"].to_numpy()

        # Normalize to be safe
        norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        norm[norm == 0.0] = 1.0
        qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

        gbx = 2.0 * (qz * qx - qy * qw)
        gby = 2.0 * (qz * qy + qx * qw)
        gbz = 1.0 - 2.0 * (qx * qx + qy * qy)

        df = df.copy()
        df["gbx"], df["gby"], df["gbz"] = gbx, gby, gbz
        return df


def plot_angrate_and_gravity(df: pd.DataFrame, out_path: str | None = None, show: bool = False) -> str | None:
    # Zero time so t0 = 0 at beginning
    t0 = df["time"].to_numpy()
    t0 = t0 - t0[0]

    # 2x3 subplots: top=wx,wy,wz; bottom=gbx->gx,gby->gy,gbz->gz
    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)

    # Angular rates
    for i, (col, ttl) in enumerate(zip(["wx", "wy", "wz"], ["wx", "wy", "wz"])):
        ax = axes[0, i]
        ax.plot(t0, df[col], label=ttl)
        ax.axvline(0.0, color="k", lw=0.8, alpha=0.4)
        ax.set_title(ttl)
        ax.set_ylabel("rad/s" if i == 0 else "")
        ax.grid(True, alpha=0.3)

    # Projected gravity components (rename for plot as gx,gy,gz)
    for i, (col, ttl) in enumerate(zip(["gbx", "gby", "gbz"], ["gx", "gy", "gz"])):
        ax = axes[1, i]
        ax.plot(t0, df[col], label=ttl)
        ax.axhline(0.0, color="k", lw=0.6, alpha=0.25)
        ax.axvline(0.0, color="k", lw=0.8, alpha=0.4)
        ax.set_title(ttl + " (unit)")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("unit" if i == 0 else "")
        ax.grid(True, alpha=0.3)

    fig.suptitle("IMU angular rates and projected gravity (g=[0,0,1] in base frame)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    saved = None
    if out_path is None:
        out_dir = os.path.join(os.path.dirname(__file__), "out")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "imu_gyro_gravity.png")
    try:
        fig.savefig(out_path, dpi=150)
        saved = out_path
    except Exception:
        saved = None

    if show:
        plt.show()
    else:
        plt.close(fig)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Read quaternion and angular rate from IMU CSV")
    parser.add_argument(
        "--csv",
        default=os.path.join(os.path.dirname(__file__), "csv_out", "common_group_decoded.csv"),
        help="Path to decoded VectorNav CSV",
    )
    parser.add_argument("--show", action="store_true", help="Display plots interactively (also saves PNG)")
    parser.add_argument("--out", default=None, help="Optional output PNG path for the figure")
    args = parser.parse_args()

    df = load_imu_quat_angrate(args.csv)
    df = compute_g_in_base(df)

    print("Rows:", len(df), "Cols:", list(df.columns))
    print("First rows with gravity in base (gbx,gby,gbz):")
    print(df[["time", "wx", "wy", "wz", "gbx", "gby", "gbz"]].head(10))

    saved_path = plot_angrate_and_gravity(df, out_path=args.out, show=args.show)
    if saved_path:
        print(f"Saved figure to: {saved_path}")


if __name__ == "__main__":
    main()
