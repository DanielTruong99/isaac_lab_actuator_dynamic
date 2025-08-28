import sys
import os
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


LEFT_JOINTS: List[str] = [
    'L_hip_joint',
    'L_hip2_joint',
    'L_thigh_joint',
    'L_calf_joint',
    'L_toe_joint',
]


def _load_and_validate(csv_path: str, normalize_time: bool = True) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"topic", "time_sec", "joint", "position"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    # effort is optional but used if available
    if normalize_time:
        try:
            t0 = float(df['time_sec'].min())
            if np.isfinite(t0):
                df['time_sec'] = df['time_sec'] - t0
        except Exception:
            pass
    return df


def plot_joint_vs_cmd(
    csv_path: str,
    joint_name: str = 'L_toe_joint',
    state_topic: str = '/joint_states',
    cmd_topic: str = '/filtered_joint_cmd',
    out_png: str | None = None,
):
    df = _load_and_validate(csv_path)

    # Filter selections
    state_df = df[(df['topic'] == state_topic) & (df['joint'] == joint_name)][['time_sec', 'position']].copy()
    # For cmd, also keep effort if present
    cmd_cols = ['time_sec', 'position'] + ([ 'effort' ] if 'effort' in df.columns else [])
    cmd_df = df[(df['topic'] == cmd_topic) & (df['joint'] == joint_name)][cmd_cols].copy()

    # Sort by time
    state_df = state_df.sort_values('time_sec')
    cmd_df = cmd_df.sort_values('time_sec')

    if state_df.empty and cmd_df.empty:
        print(f"No data found for joint '{joint_name}' in either topic.")
        return

    fig, ax_pos = plt.subplots(figsize=(12, 6))
    if not state_df.empty:
        ax_pos.plot(state_df['time_sec'], state_df['position'], label=f"{joint_name} (state)")
    if not cmd_df.empty:
        ax_pos.plot(cmd_df['time_sec'], cmd_df['position'], label=f"{joint_name} (cmd)")

    ax_pos.set_title(f"{joint_name}: state vs cmd + computed pd torque")
    ax_pos.set_xlabel('Time (s)')
    ax_pos.set_ylabel('Position (rad)')
    ax_pos.grid(True, alpha=0.3)

    # Plot torque on twin axis if available
    if 'effort' in cmd_df.columns and not cmd_df.empty and cmd_df['effort'].notna().any():
        ax_tau = ax_pos.twinx()
        ax_tau.plot(cmd_df['time_sec'], cmd_df['effort'], color='tab:red', alpha=0.7, label='computed pd torque')
        ax_tau.set_ylabel('Torque (Nm)')
        # Build combined legend
        lines, labels = ax_pos.get_legend_handles_labels()
        lines2, labels2 = ax_tau.get_legend_handles_labels()
        ax_pos.legend(lines + lines2, labels + labels2, loc='upper right')
    else:
        ax_pos.legend(loc='upper right')

    plt.tight_layout()

    if out_png is None:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        safe_joint = joint_name.replace('/', '_')
        out_png = f"{base}_{safe_joint}_state_cmd_tau.png"
    # fig.savefig(out_png, dpi=150)
    plt.show()
    # print(f"Saved plot to {out_png}")


def plot_left_joints_grid(
    csv_path: str,
    joints: List[str] | None = None,
    state_topic: str = '/joint_states',
    cmd_topic: str = '/filtered_joint_cmd',
    out_png: str | None = None,
):
    df = _load_and_validate(csv_path)
    joints = joints or LEFT_JOINTS

    n = len(joints)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.6 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax_pos, joint_name in zip(axes, joints):
        # Filter and sort
        state_cols = ['time_sec', 'position'] + (['effort'] if 'effort' in df.columns else [])
        state_df = df[(df['topic'] == state_topic) & (df['joint'] == joint_name)][state_cols].copy()
        cmd_cols = ['time_sec', 'position'] + ([ 'effort' ] if 'effort' in df.columns else [])
        cmd_df = df[(df['topic'] == cmd_topic) & (df['joint'] == joint_name)][cmd_cols].copy()
        state_df = state_df.sort_values('time_sec')
        cmd_df = cmd_df.sort_values('time_sec')

        if state_df.empty and cmd_df.empty:
            ax_pos.text(0.5, 0.5, f"No data for {joint_name}", transform=ax_pos.transAxes, ha='center')
            continue

        if not state_df.empty:
            ax_pos.plot(state_df['time_sec'], state_df['position'], label='state pos')
        if not cmd_df.empty:
            ax_pos.plot(cmd_df['time_sec'], cmd_df['position'], label='cmd pos')
        ax_pos.set_ylabel('pos (rad)')
        ax_pos.set_title(joint_name)
        ax_pos.grid(True, alpha=0.3)

        # Effort on twin axis if present
        has_effort = False
        if 'effort' in df.columns:
            ax_tau = ax_pos.twinx()
            
            # Plot state effort if available
            if 'effort' in state_df.columns and not state_df.empty and state_df['effort'].notna().any():
                ax_tau.plot(state_df['time_sec'], state_df['effort'], color='tab:blue', alpha=0.5, label='state effort')
                has_effort = True
                
            # Plot cmd effort if available    
            if 'effort' in cmd_df.columns and not cmd_df.empty and cmd_df['effort'].notna().any():
                ax_tau.plot(cmd_df['time_sec'], cmd_df['effort'], color='tab:red', alpha=0.7, label='cmd effort')
                has_effort = True
                
            if has_effort:
                ax_tau.set_ylabel('effort (Nm)')
                # Legend per subplot
                lines, labels = ax_pos.get_legend_handles_labels()
                lines2, labels2 = ax_tau.get_legend_handles_labels()
                ax_pos.legend(lines + lines2, labels + labels2, loc='upper right', fontsize='small')
            else:
                ax_tau.remove()
                ax_pos.legend(loc='upper right', fontsize='small')
        else:
            ax_pos.legend(loc='upper right', fontsize='small')

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout(h_pad=0.4)

    if out_png is None:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        out_png = f"{base}_L_left_all_state_cmd_tau.png"
    plt.show(fig)


def plot_left_histograms_window(
    csv_path: str,
    t0: float = 20.0,
    t1: float = 25.0,
    joints: List[str] | None = None,
    topic: str = '/joint_states',
    bins: int = 60,
    out_png: str | None = None,
):
    df = _load_and_validate(csv_path)
    joints = joints or LEFT_JOINTS

    # Filter by time window and topic
    mask = (df['topic'] == topic) & (df['time_sec'] >= t0) & (df['time_sec'] <= t1)
    cols = ['joint', 'position', 'velocity'] + (['effort'] if 'effort' in df.columns else [])
    wdf = df.loc[mask, cols].copy()

    if wdf.empty:
        print(f"No data in topic '{topic}' between t=[{t0}, {t1}] s")
        return

    n = len(joints)
    ncols = 3 if 'effort' in wdf.columns else 2
    fig, axes = plt.subplots(n, ncols, figsize=(7 * ncols, 2.4 * n), sharex=False)
    if n == 1:
        axes = np.array([axes])
    if ncols == 2 and n > 1:
        pass  # axes is already (n, 2)
    elif ncols == 3 and n == 1:
        axes = axes.reshape(1, 3)
    elif ncols == 2 and n == 1:
        axes = axes.reshape(1, 2)

    # Collect stats
    stats_rows = []

    for i, jname in enumerate(joints):
        jdf = wdf[wdf['joint'] == jname]
        # Position histogram
        axp = axes[i, 0]
        pos = jdf['position'].dropna().values
        if pos.size > 0:
            axp.hist(pos, bins=bins, color='tab:blue', alpha=0.8, edgecolor='white')
            p_mean = float(np.mean(pos))
            p_std = float(np.std(pos))
        else:
            p_mean = float('nan')
            p_std = float('nan')
        axp.set_title(f"{jname} pos (rad)")
        axp.set_ylabel('count')
        axp.grid(True, alpha=0.2)
        # Annotate with stats
        axp.text(0.98, 0.95, f"μ={p_mean:.3f}\nσ={p_std:.3f}", transform=axp.transAxes,
                 ha='right', va='top', fontsize=9,
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

        # Velocity histogram
        axv = axes[i, 1]
        if 'velocity' in jdf.columns:
            vel = jdf['velocity'].dropna().values
        else:
            vel = np.array([])
        if vel.size > 0:
            axv.hist(vel, bins=bins, color='tab:green', alpha=0.8, edgecolor='white')
            v_mean = float(np.mean(vel))
            v_std = float(np.std(vel))
        else:
            v_mean = float('nan')
            v_std = float('nan')
        axv.set_title(f"{jname} vel (rad/s)")
        axv.grid(True, alpha=0.2)
        axv.text(0.98, 0.95, f"μ={v_mean:.3f}\nσ={v_std:.3f}", transform=axv.transAxes,
                 ha='right', va='top', fontsize=9,
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

        # Effort histogram (if available)
        if ncols == 3:
            axe = axes[i, 2]
            if 'effort' in jdf.columns:
                eff = jdf['effort'].dropna().values
            else:
                eff = np.array([])
            if eff.size > 0:
                axe.hist(eff, bins=bins, color='tab:orange', alpha=0.8, edgecolor='white')
                e_mean = float(np.mean(eff))
                e_std = float(np.std(eff))
            else:
                e_mean = float('nan')
                e_std = float('nan')
            axe.set_title(f"{jname} effort (Nm)")
            axe.grid(True, alpha=0.2)
            axe.text(0.98, 0.95, f"μ={e_mean:.3f}\nσ={e_std:.3f}", transform=axe.transAxes,
                     ha='right', va='top', fontsize=9,
                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        else:
            e_mean = float('nan')
            e_std = float('nan')
            eff = np.array([])

        stats_rows.append({
            'joint': jname,
            'pos_mean': p_mean,
            'pos_std': p_std,
            'pos_count': int(pos.size),
            'vel_mean': v_mean,
            'vel_std': v_std,
            'vel_count': int(vel.size),
            'eff_mean': e_mean,
            'eff_std': e_std,
            'eff_count': int(eff.size),
        })

    # Common labels and layout
    fig.suptitle(f"Left joints histograms, {t0:.2f}–{t1:.2f}s ({topic})", y=0.995)
    for ax in axes[-1, :]:
        ax.set_xlabel('value')
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if out_png is None:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        out_png = f"{base}_left_hist_{int(t0)}_{int(t1)}.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"Saved histograms to {out_png}")

    # Save stats CSV next to figure
    try:
        base = os.path.splitext(out_png)[0] if out_png else os.path.splitext(os.path.basename(csv_path))[0]
        stats_path = f"{base}_stats.csv"
        pd.DataFrame(stats_rows).to_csv(stats_path, index=False)
        print(f"Saved stats to {stats_path}")
    except Exception as e:
        print(f"Failed to save stats CSV: {e}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 plot_joint_vs_cmd.py <joint_states.csv> [joint_name|ALL_LEFT|HIST_LEFT] [out.png|t0 t1 [out.png]]')
        print('Note: time_sec is normalized to start at 0 (global min subtracted). For HIST_LEFT, t0/t1 are in normalized seconds.')
        sys.exit(1)
    csv = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else 'ALL_LEFT'

    if mode.upper() in ('ALL_LEFT', 'ALL_L', 'LEFT'):
        out = sys.argv[3] if len(sys.argv) > 3 else None
        plot_left_joints_grid(csv, out_png=out)
    elif mode.upper() in ('HIST_LEFT', 'HIST'):
        # Optional t0 t1 out
        if len(sys.argv) >= 5:
            try:
                t0 = float(sys.argv[3])
                t1 = float(sys.argv[4])
            except ValueError:
                print('Invalid t0/t1; expected numbers. Falling back to 20 25.')
                t0, t1 = 20.0, 25.0
            out = sys.argv[5] if len(sys.argv) > 5 else None
        else:
            t0, t1 = 20.0, 25.0
            out = sys.argv[3] if len(sys.argv) > 3 else None
        plot_left_histograms_window(csv, t0=t0, t1=t1, out_png=out)
    else:
        out = sys.argv[3] if len(sys.argv) > 3 else None
        plot_joint_vs_cmd(csv, joint_name=mode, out_png=out)
