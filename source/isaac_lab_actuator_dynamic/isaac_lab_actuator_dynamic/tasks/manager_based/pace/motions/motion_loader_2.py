# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import os
import torch
from typing import Optional


class MotionLoader:
    """
    Helper class to load and sample motion data from NumPy-file format.
    """

    def __init__(self, motion_file: str, device: torch.device) -> None:
        """Load a motion file and initialize the internal variables.

        Args:
            motion_file: Motion file path to load.
            device: The device to which to load the data.

        Raises:
            AssertionError: If the specified motion file doesn't exist.
        """
        # Support multiple motion files
        if isinstance(motion_file, (list, tuple)):
            data_list = []
            fps = []
            dof_names = []
            dof_positions = []
            dof_velocities = []
            dof_efforts = []
            dof_position_commands = []
            motion_dt = []
            motion_duration = []
            motion_num_frames = []
            for index, mf in enumerate(motion_file):
                data_list.append(np.load(mf, allow_pickle=True))
                dof_names.append(data_list[index]["dof_names"].tolist())
                fps.append(data_list[index]["fps"])
                dof_positions.append(data_list[index]["dof_positions"])
                dof_velocities.append(data_list[index]["dof_velocities"])
                dof_efforts.append(data_list[index]["dof_efforts"])
                dof_position_commands.append(data_list[index]["dof_position_commands"])

                motion_num_frames.append(data_list[index]["dof_positions"].shape[0])
                motion_dt.append(1.0 / fps[index])
                motion_duration.append(motion_dt[index] * (motion_num_frames[index] - 1))
            dof_names = dof_names[0]
        else:
            assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
            data = np.load(motion_file)
            dof_names = data["dof_names"].tolist()
            fps = data["fps"]
            dof_positions = data["dof_positions"]
            dof_velocities = data["dof_velocities"]
            dof_efforts = dof_positions
            dof_position_commands = dof_positions

        self.device = device
        self._dof_names = dof_names

        self.dof_positions = torch.from_numpy(np.concatenate(dof_positions, axis=0)).to(device=self.device, dtype=torch.float32)
        self.dof_velocities = torch.from_numpy(np.concatenate(dof_velocities, axis=0)).to(device=self.device, dtype=torch.float32)
        self.dof_efforts = torch.from_numpy(np.concatenate(dof_efforts, axis=0)).to(device=self.device, dtype=torch.float32)
        self.dof_position_commands = torch.from_numpy(np.concatenate(dof_position_commands, axis=0)).to(device=self.device, dtype=torch.float32)

        #! Temporary fix
        # self._dof_names = self._dof_names[:5]
        self.dof_positions = self.dof_positions[:, :5]
        self.dof_velocities = self.dof_velocities[:, :5]
        self.dof_efforts = self.dof_efforts[:, :5]
        self.dof_position_commands = self.dof_position_commands[:, :5]

        self.dt = torch.tensor(motion_dt, dtype=torch.float32, device=self.device)
        self.num_frames = torch.tensor(motion_num_frames, dtype=torch.int32, device=self.device)
        self.duration = torch.tensor(motion_duration, dtype=torch.float32, device=self.device)
        self.num_motions = self.num_frames.shape[0]

        self.start_motion_ids = self.num_frames.roll(1)
        self.start_motion_ids[0] = 0
        self.start_motion_ids = self.start_motion_ids.cumsum(0)

        print(f"Motion loaded: duration: {self.duration.sum()} sec, frames: {self.num_frames.sum()}")

    @property
    def dof_names(self) -> list[str]:
        """Skeleton DOF names."""
        return self._dof_names


    @property
    def num_dofs(self) -> int:
        """Number of skeleton's DOFs."""
        return len(self._dof_names)


    def _interpolate(
        self,
        a: torch.Tensor,
        *,
        b: Optional[torch.Tensor] = None,
        blend: Optional[torch.Tensor] = None,
        start: Optional[np.ndarray] = None,
        end: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Linear interpolation between consecutive values.

        Args:
            a: The first value. Shape is (N, X) or (N, M, X).
            b: The second value. Shape is (N, X) or (N, M, X).
            blend: Interpolation coefficient between 0 (a) and 1 (b).
            start: Indexes to fetch the first value. If both, ``start`` and ``end` are specified,
                the first and second values will be fetches from the argument ``a`` (dimension 0).
            end: Indexes to fetch the second value. If both, ``start`` and ``end` are specified,
                the first and second values will be fetches from the argument ``a`` (dimension 0).

        Returns:
            Interpolated values. Shape is (N, X) or (N, M, X).
        """
        if start is not None and end is not None:
            return self._interpolate(a=a[start], b=a[end], blend=blend)
        if a.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if a.ndim >= 3:
            blend = blend.unsqueeze(-1)
        return (1.0 - blend) * a + blend * b

    def _slerp(
        self,
        q0: torch.Tensor,
        *,
        q1: Optional[torch.Tensor] = None,
        blend: Optional[torch.Tensor] = None,
        start: Optional[np.ndarray] = None,
        end: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Interpolation between consecutive rotations (Spherical Linear Interpolation).

        Args:
            q0: The first quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            q1: The second quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            blend: Interpolation coefficient between 0 (q0) and 1 (q1).
            start: Indexes to fetch the first quaternion. If both, ``start`` and ``end` are specified,
                the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).
            end: Indexes to fetch the second quaternion. If both, ``start`` and ``end` are specified,
                the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).

        Returns:
            Interpolated quaternions. Shape is (N, 4) or (N, M, 4).
        """
        if start is not None and end is not None:
            return self._slerp(q0=q0[start], q1=q0[end], blend=blend)
        if q0.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if q0.ndim >= 3:
            blend = blend.unsqueeze(-1)

        qw, qx, qy, qz = 0, 1, 2, 3  # wxyz
        cos_half_theta = (
            q0[..., qw] * q1[..., qw]
            + q0[..., qx] * q1[..., qx]
            + q0[..., qy] * q1[..., qy]
            + q0[..., qz] * q1[..., qz]
        )

        neg_mask = cos_half_theta < 0
        q1 = q1.clone()
        q1[neg_mask] = -q1[neg_mask]
        cos_half_theta = torch.abs(cos_half_theta)
        cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

        half_theta = torch.acos(cos_half_theta)
        sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

        ratio_a = torch.sin((1 - blend) * half_theta) / sin_half_theta
        ratio_b = torch.sin(blend * half_theta) / sin_half_theta

        new_q_x = ratio_a * q0[..., qx : qx + 1] + ratio_b * q1[..., qx : qx + 1]
        new_q_y = ratio_a * q0[..., qy : qy + 1] + ratio_b * q1[..., qy : qy + 1]
        new_q_z = ratio_a * q0[..., qz : qz + 1] + ratio_b * q1[..., qz : qz + 1]
        new_q_w = ratio_a * q0[..., qw : qw + 1] + ratio_b * q1[..., qw : qw + 1]

        new_q = torch.cat([new_q_w, new_q_x, new_q_y, new_q_z], dim=len(new_q_w.shape) - 1)
        new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
        new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)
        return new_q

    def _compute_frame_blend(self, time, len, num_frames, dt):
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = (time - frame_idx0 * dt) / dt

        return frame_idx0, frame_idx1, blend

    def sample_times(self, motion_ids):
        """Sample random motion times uniformly.

        Args:
            num_samples: Number of time samples to generate.
            duration: Maximum motion duration to sample.
                If not defined samples will be within the range of the motion duration.

        Raises:
            AssertionError: If the specified duration is longer than the motion duration.

        Returns:
            Time samples, between 0 and the specified/motion duration.
        """
        phase = torch.rand(motion_ids.shape, device=self.device)
        duration = self.duration[motion_ids]
        return duration * phase

    def sample(self, motion_ids, motion_times):
        motion_len = self.duration[motion_ids]
        num_frames = self.num_frames[motion_ids]
        dt = self.dt[motion_ids]

        index_0, index_1, blend = self._compute_frame_blend(motion_times, motion_len, num_frames, dt)
        index_0 = index_0 + self.start_motion_ids[motion_ids]
        index_1 = index_1 + self.start_motion_ids[motion_ids]

        return (
            self._interpolate(self.dof_positions, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.dof_velocities, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.dof_position_commands, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.dof_efforts, blend=blend, start=index_0, end=index_1),
        )

    def get_dof_index(self, dof_names: list[str]) -> list[int]:
        """Get skeleton DOFs indexes by DOFs names.

        Args:
            dof_names: List of DOFs names.

        Raises:
            AssertionError: If the specified DOFs name doesn't exist.

        Returns:
            List of DOFs indexes.
        """
        indexes = []
        for name in dof_names:
            assert name in self._dof_names, f"The specified DOF name ({name}) doesn't exist: {self._dof_names}"
            indexes.append(self._dof_names.index(name))
        return indexes



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Motion file")
    args, _ = parser.parse_known_args()

    motion = MotionLoader(args.file, "cpu")

    print("- number of frames:", motion.num_frames)
    print("- number of DOFs:", motion.num_dofs)
