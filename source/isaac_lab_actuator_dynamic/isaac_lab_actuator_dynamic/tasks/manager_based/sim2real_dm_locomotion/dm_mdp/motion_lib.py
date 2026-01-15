import os
from copy import deepcopy
from typing import Any, Tuple
from easydict import EasyDict

import numpy as np
import torch
import yaml
from torch import Tensor, nn
import copy

import sys



class LoadedMotions(nn.Module):
    def __init__(
            self,
            motions,
            motion_lengths: Tensor,
            motion_weights: Tensor,
            motion_fps: Tensor,
            motion_dt: Tensor,
            motion_num_frames: Tensor,
            motion_files: Tuple[str],
            ref_respawn_offsets: Tensor,
            **kwargs,  # Catch some nn.Module arguments that aren't needed
    ):
        super().__init__()
        self.motions = motions
        self.motion_files = motion_files
        self.register_buffer("motion_lengths", motion_lengths, persistent=False)
        self.register_buffer("motion_weights", motion_weights, persistent=False)
        self.register_buffer("motion_fps", motion_fps, persistent=False)
        self.register_buffer("motion_dt", motion_dt, persistent=False)
        self.register_buffer("motion_num_frames", motion_num_frames, persistent=False)
        self.register_buffer(
            "ref_respawn_offsets", ref_respawn_offsets, persistent=False
        )


class MotionLib(DeviceDtypeModuleMixin):
    gts: Tensor
    grs: Tensor
    lrs: Tensor
    gvs: Tensor
    gavs: Tensor
    grvs: Tensor
    gravs: Tensor
    dvs: Tensor
    length_starts: Tensor
    motion_ids: Tensor
    key_body_ids: Tensor

    def __init__(
            self,
            motion_file,
            robot_config: RobotConfig,
            key_body_ids,
            device="cpu",
            ref_height_adjust: float = 0,
            target_frame_rate: int = 30,
            create_text_embeddings: bool = False,
            fix_motion_heights: bool = True,
            skeleton_tree: Any = None,
            local_rot_conversion: Tensor = None,
            w_last: bool = True,
    ):
        super().__init__()
        self.w_last = w_last
        self.fix_heights = fix_motion_heights
        self.skeleton_tree = skeleton_tree
        self.create_text_embeddings = create_text_embeddings
        self.robot_config = robot_config
        self.ref_height_adjust = ref_height_adjust
        self.local_rot_conversion = local_rot_conversion

        self.register_buffer(
            "key_body_ids",
            torch.tensor(key_body_ids, dtype=torch.long, device=device),
            persistent=False,
        )

        if str(motion_file).split(".")[-1] in ["yaml", "npy", "npz", "np"]:
            print("Loading motions from yaml/npy file")
            self._load_motions(motion_file, target_frame_rate)

        self.motion_file = motion_file

        motions = self.state.motions
        self.register_buffer(
            "gts",
            torch.cat([m.global_translation for m in motions], dim=0).to(
                dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "grs",
            torch.cat([m.global_rotation for m in motions], dim=0).to(
                dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "lrs",
            torch.cat([m.local_rotation for m in motions], dim=0).to(
                dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "grvs",
            torch.cat([m.global_root_velocity for m in motions], dim=0).to(
                dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "gravs",
            torch.cat([m.global_root_angular_velocity for m in motions], dim=0).to(
                dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "gavs",
            torch.cat([m.global_angular_velocity for m in motions], dim=0).to(
                dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "gvs",
            torch.cat([m.global_velocity for m in motions], dim=0).to(
                dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "dvs",
            torch.cat([m.dof_vels for m in motions], dim=0).to(
                device=device, dtype=torch.float32
            ),
            persistent=False,
        )

        lengths = self.state.motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.register_buffer(
            "length_starts", lengths_shifted.cumsum(0), persistent=False
        )

        self.register_buffer(
            "motion_ids",
            torch.arange(
                len(self.state.motions), dtype=torch.long, device=self._device
            ),
            persistent=False,
        )

        self.to(device)

    def num_motions(self):
        """Returns the number of motions in the state.

        Returns:
            int: The number of motions.
        """
        return len(self.state.motions)

    def get_total_length(self):
        """Returns the total length of all motions.

        Returns:
            int: The total length of all motions.
        """
        return sum(self.state.motion_lengths)

    def get_motion(self, motion_id):
        return self.state.motions[motion_id]

    def sample_motions(self, n, valid_mask=None):
        if valid_mask is not None:
            weights = self.state.motion_weights.clone()
            weights[~valid_mask] = 0
        else:
            weights = self.state.motion_weights

        motion_ids = torch.multinomial(weights, num_samples=n, replacement=True)

        return motion_ids

    def sample_text_embeddings(self, motion_ids: Tensor) -> Tensor:
        """Samples text embeddings for the given motion IDs.

        Args:
            motion_ids (Tensor): A tensor containing the IDs of the motions.

        Returns:
            Tensor: A tensor containing the sampled text embeddings for the given motion IDs.
        """
        if hasattr(self.state, "text_embeddings"):
            indices = torch.randint(0, 3, (motion_ids.shape[0],), device=self.device)
            return self.state.text_embeddings[motion_ids, indices]
        return 0

    def sample_time(self, motion_ids, truncate_time=None):
        phase = torch.rand(motion_ids.shape, device=self.device)

        motion_len = self.state.motion_lengths[motion_ids]

        if truncate_time is not None:
            assert truncate_time >= 0.0
            motion_len -= truncate_time
            assert torch.all(motion_len >= 0)

        motion_time = phase * motion_len
        return motion_time

    def get_motion_length(self, motion_ids):
        if motion_ids is None:
            return self.state.motion_lengths
        else:
            return self.state.motion_lengths[motion_ids]

    def get_motion_state(
            self, motion_ids, motion_times, joint_3d_format="exp_map"
    ) -> RobotState:
        motion_len = self.state.motion_lengths[motion_ids]
        motion_times = motion_times.clip(min=0).clip(
            max=motion_len
        )  # Making sure time is in bounds

        num_frames = self.state.motion_num_frames[motion_ids]
        dt = self.state.motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(
            motion_times, motion_len, num_frames, dt
        )

        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        root_pos0 = self.gts[f0l, 0]
        root_pos1 = self.gts[f1l, 0]

        root_rot0 = self.grs[f0l, 0]
        root_rot1 = self.grs[f1l, 0]

        local_rot0 = self.lrs[f0l]
        local_rot1 = self.lrs[f1l]

        root_vel0 = self.grvs[f0l]
        root_vel1 = self.grvs[f1l]

        root_ang_vel0 = self.gravs[f0l]
        root_ang_vel1 = self.gravs[f1l]

        global_vel0 = self.gvs[f0l]
        global_vel1 = self.gvs[f1l]

        global_ang_vel0 = self.gavs[f0l]
        global_ang_vel1 = self.gavs[f1l]

        key_body_pos0 = self.gts[f0l.unsqueeze(-1), self.key_body_ids.unsqueeze(0)]
        key_body_pos1 = self.gts[f1l.unsqueeze(-1), self.key_body_ids.unsqueeze(0)]

        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        rigid_body_pos0 = self.gts[f0l]
        rigid_body_pos1 = self.gts[f1l]

        rigid_body_rot0 = self.grs[f0l]
        rigid_body_rot1 = self.grs[f1l]

        vals = [
            root_pos0,
            root_pos1,
            local_rot0,
            local_rot1,
            root_vel0,
            root_vel1,
            root_ang_vel0,
            root_ang_vel1,
            global_vel0,
            global_vel1,
            global_ang_vel0,
            global_ang_vel1,
            dof_vel0,
            dof_vel1,
            key_body_pos0,
            key_body_pos1,
            rigid_body_pos0,
            rigid_body_pos1,
            rigid_body_rot0,
            rigid_body_rot1,
        ]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        root_pos: Tensor = (1.0 - blend) * root_pos0 + blend * root_pos1
        root_pos[:, 2] += self.ref_height_adjust

        root_rot: Tensor = torch_utils.slerp(root_rot0, root_rot1, blend)

        blend_exp = blend.unsqueeze(-1)
        key_body_pos = (1.0 - blend_exp) * key_body_pos0 + blend_exp * key_body_pos1
        key_body_pos[:, :, 2] += self.ref_height_adjust

        if hasattr(self, "dof_pos"):  # H1 joints
            dof_pos = (1.0 - blend) * self.dof_pos[f0l] + blend * self.dof_pos[f1l]
        else:
            local_rot = torch_utils.slerp(
                local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1)
            )
            dof_pos: Tensor = self._local_rotation_to_dof(local_rot, joint_3d_format)

        root_vel = (1.0 - blend) * root_vel0 + blend * root_vel1
        root_ang_vel = (1.0 - blend) * root_ang_vel0 + blend * root_ang_vel1
        dof_vel = (1.0 - blend) * dof_vel0 + blend * dof_vel1
        rigid_body_pos = (1.0 - blend_exp) * rigid_body_pos0 + blend_exp * rigid_body_pos1
        rigid_body_pos[:, :, 2] += self.ref_height_adjust
        rigid_body_rot = torch_utils.slerp(rigid_body_rot0, rigid_body_rot1, blend_exp)
        global_vel = (1.0 - blend_exp) * global_vel0 + blend_exp * global_vel1
        global_ang_vel = (
            1.0 - blend_exp
        ) * global_ang_vel0 + blend_exp * global_ang_vel1

        motion_state = RobotState(
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            key_body_pos=key_body_pos,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            rigid_body_pos=rigid_body_pos,
            rigid_body_rot=rigid_body_rot,
            rigid_body_vel=global_vel,
            rigid_body_ang_vel=global_ang_vel,
        )

        return motion_state

    @staticmethod
    def _load_motion_file(self, motion_file):
        '''
            npz file from loco mujoco framework includes:
            - qpos: T x J, Position of the joints, including the root joint.
            - qvel: T x J, Velocity of the joints, including the root joint.
            - xpos: T x B x 3, Position of all bodies in global coordinates.
            - xquat: T x B x 4, Quaternion of all bodies in global coordinates.
            - cvel: T x B x 6, Velocity of all bodies in global coordinates, v + w.
            - site_xpos: T x S x 3, Position of all sites in global coordinates, mimic.
            - site_xquat: T x S x 4, Quaternion of all sites in global coordinates, mimic.
            - joint_names: List of joint names.
            - body_names: List of body names.
            - site_names: List of site names.
            - frequency: Frequency of the motion data.
            - njnts: Number of joints.
            - split_points: List of split points for the motion data.

            npy file from h1_walk.npy includes:
            - dof_pos: T x J, Position of the joints, including the root joint.
            - dof_vel: T x J, Velocity of the joints, including the root joint.
            - fps: Frequency of the motion data.
            - global_angular_velocity: T x B x 3, Angular velocity of all bodies in global coordinates.
            - global_root_angular_velocity: T x 3, Angular velocity of the root joint.
            - global_root_velocity: T x 3, Linear velocity of the root joint.
            - global_rotation: T x B x 4, Quaternion of all bodies in global coordinates.
            - global_rotation_mat: T x B x 3 x 3, Rotation matrix of all bodies in global coordinates.
            - global_translation: T x B x 3, Position of all bodies in global coordinates.
            - global_velocity: T x B x 6, Velocity of all bodies in global coordinates, v + w.
            - local_rotation: T x J x 3, Local rotation of the joints.
        '''

        if motion_file.endswith(".npy"):
            motion = EasyDict(torch.load(motion_file))
        elif motion_file.endswith(".npz"):
            motion_data = np.load(motion_file, allow_pickle=True)
            motion = EasyDict()
            for key in motion_data.files:
                if isinstance(motion_data[key], np.ndarray) and np.issubdtype(motion_data[key].dtype, np.number):
                    motion[key] = torch.tensor(motion_data[key])
                else:
                    motion[key] = motion_data[key]  
            
            
            motion.body_names = np.delete(motion.body_names, motion.body_names == 'world') # Remove the world body
            motion.body_names[motion.body_names == 'root'] = 'Pelvis'  # Rename root to Pelvis

            # Map member names to match the expected structure
            motion.dof_pos = motion.qpos[:, 7:]  # Exclude the root
            motion.dof_vels = motion.qvel[:, 6:]  # Exclude the root
            motion.fps = int(motion.frequency)
            motion.global_angular_velocity = motion.cvel[:, 1:, 3:]  # Exclude the root
            motion.global_root_angular_velocity = motion.cvel[:, 1, 3:] # Pelvis is the root
            motion.global_root_velocity = motion.cvel[:, 1, :3]  # Pelvis is the root
            motion.global_rotation = motion.xquat[:, 1:, [1, 2, 3, 0]]  # Exclude the root
            motion.global_rotation_mat = torch.stack(
                [quaternion_to_matrix(q, w_last=True) for q in motion.global_rotation], dim=0
            )
            motion.global_translation = motion.xpos[:, 1:]  # Exclude the root
            motion.global_velocity = motion.cvel[:, 1:, :3]  # Exclude the root

        return motion

    @staticmethod
    def _slice_motion_file(motion, motion_timings):
        start, end = motion_timings
        start_frame = round(start * motion.fps)
        if end == -1:
            end_frame = motion.global_translation.shape[0]
        else:
            end_frame = int(end * motion.fps)

        assert (
                start_frame < end_frame
        ), f"Motion start frame {start_frame} >= motion end frame {end_frame}"

        sliced_motion = {}

        for key in motion.keys():
            # if is torch.Tensor
            if isinstance(motion[key], torch.Tensor):
                if motion[key].ndim < 2: continue
                if motion[key].shape[0] < end_frame: continue
                sliced_motion[key] = motion[key][start_frame:end_frame].clone()
            else:
                sliced_motion[key] = copy.deepcopy(motion[key])

        return EasyDict(sliced_motion)

    def _load_motions(self, motion_file, target_frame_rate):
        if self.create_text_embeddings:
            from transformers import AutoTokenizer, XCLIPTextModel

            model = XCLIPTextModel.from_pretrained("microsoft/xclip-base-patch32")
            tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")

        motions = []
        motion_lengths = []
        motion_dt = []
        motion_num_frames = []
        text_embeddings = []
        has_text_embeddings = []
        motion_fpses = []
        (
            motion_files,
            motion_weights,
            motion_timings,
            full_motion_fpses,
            sub_motion_to_motion,
            ref_respawn_offsets,
            motion_labels,
        ) = self._fetch_motion_files(motion_file)

        num_sub_motions = len(sub_motion_to_motion)

        for f in range(num_sub_motions):
            motion_f = sub_motion_to_motion[f]
            curr_file = motion_files[motion_f]
            print(
                "Loading {:d}/{:d} motion files: {:s}".format(
                    f + 1, num_sub_motions, curr_file
                )
            )

            curr_motion = self._load_motion_file(curr_file)

            cur_fps = full_motion_fpses[motion_f]
            if cur_fps is None:
                cur_fps = curr_motion.fps
                
            if cur_fps > target_frame_rate:
                # Not necessary, but we downsample the FPS to save memory
                # do nothing if cur_fps <= target_frame_rate
                curr_motion = self._fix_motion_fps(
                    curr_motion,
                    cur_fps,
                    target_frame_rate,
                    self.skeleton_tree,
                )

            sub_motion = self._slice_motion_file(curr_motion, motion_timings[f])
            motion_fpses.append(float(sub_motion.fps))

            if self.fix_heights:
                sub_motion = self.fix_motion_heights(sub_motion, self.skeleton_tree)

            curr_dt = 1.0 / motion_fpses[f]

            num_frames = sub_motion.global_translation.shape[0]
            curr_len = 1.0 / motion_fpses[f] * (num_frames - 1)

            motion_dt.append(curr_dt)
            motion_num_frames.append(num_frames)

            curr_dof_vels = self._compute_motion_dof_vels(sub_motion)
            sub_motion.dof_vels = curr_dof_vels

            motions.append(sub_motion)
            motion_lengths.append(curr_len)

            if self.create_text_embeddings and motion_labels[f][0] != "":
                with torch.inference_mode():
                    inputs = tokenizer(
                        motion_labels[f],
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    )
                    outputs = model(**inputs)
                    pooled_output = outputs.pooler_output  # pooled (EOS token) states
                    text_embeddings.append(pooled_output)  # should be [3, 512]
                    has_text_embeddings.append(True)
            else:
                text_embeddings.append(
                    torch.zeros((3, 512), dtype=torch.float32)
                )  # just hold something temporary
                has_text_embeddings.append(False)

        motion_lengths = torch.tensor(
            motion_lengths, device=self._device, dtype=torch.float32
        )

        motion_weights = torch.tensor(
            motion_weights, dtype=torch.float32, device=self._device
        )
        motion_weights /= motion_weights.sum()

        ref_respawn_offsets = torch.tensor(
            ref_respawn_offsets, dtype=torch.float32, device=self._device
        )

        motion_fpses = torch.tensor(
            motion_fpses, device=self._device, dtype=torch.float32
        )
        motion_dt = torch.tensor(motion_dt, device=self._device, dtype=torch.float32)
        motion_num_frames = torch.tensor(motion_num_frames, device=self._device)

        text_embeddings = torch.stack(text_embeddings).detach().to(device=self._device)
        has_text_embeddings = torch.tensor(
            has_text_embeddings, dtype=torch.bool, device=self._device
        )

        self.state = LoadedMotions(
            motions=tuple(motions),
            motion_lengths=motion_lengths,
            motion_weights=motion_weights,
            motion_fps=motion_fpses,
            motion_dt=motion_dt,
            motion_num_frames=motion_num_frames,
            motion_files=tuple(motion_files),
            ref_respawn_offsets=ref_respawn_offsets,
            text_embeddings=text_embeddings,
            has_text_embeddings=has_text_embeddings,
        )

        num_motions = self.num_motions()
        total_len = self.get_total_length()

        print(
            "Loaded {:d} motions with a total length of {:.3f}s.".format(
                num_motions, total_len
            )
        )

    def _fetch_motion_files(self, motion_file):
        ext = os.path.splitext(motion_file)[1]
        if ext == ".yaml":
            dir_name = os.path.dirname(motion_file)
            motion_files = []
            sub_motion_to_motion = []
            ref_respawn_offsets = []
            motion_weights = []
            motion_timings = []
            motion_fpses = []
            motion_labels = []
            with open(os.path.join(os.getcwd(), motion_file), "r") as f:
                motion_config = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))

            motion_list = sorted(
                motion_config.motions,
                key=lambda x: 1e6 if "idx" not in x else int(x.idx),
            )

            motion_index = 0

            for motion_id, motion_entry in enumerate(motion_list):
                curr_file = motion_entry.file
                curr_file = os.path.join(dir_name, curr_file)
                motion_files.append(curr_file)
                motion_fpses.append(motion_entry.get("fps", None))

                if "sub_motions" not in motion_entry:
                    motion_entry.sub_motions = [deepcopy(motion_entry)]
                    motion_entry.sub_motions[0].idx = motion_index

                for sub_motion in sorted(
                        motion_entry.sub_motions, key=lambda x: int(x.idx)
                ):
                    curr_weight = sub_motion.weight
                    assert curr_weight >= 0

                    assert motion_index == sub_motion.idx

                    motion_weights.append(curr_weight)

                    sub_motion_to_motion.append(motion_id)

                    ref_respawn_offset = sub_motion.get("ref_respawn_offset", 0)
                    ref_respawn_offsets.append(ref_respawn_offset)

                    if "timings" in sub_motion:
                        curr_timing = sub_motion.timings
                        start = curr_timing.start
                        end = curr_timing.end
                    else:
                        start = 0
                        end = -1

                    motion_timings.append([start, end])

                    sub_motion_labels = []
                    if "labels" in sub_motion:
                        # We assume 3 labels for each motion.
                        # If there are fewer than 3 labels, the last label is repeated to fill the list.
                        # If there are no labels, an empty string is used as the label.
                        for label in sub_motion.labels:
                            sub_motion_labels.append(label)
                            if len(sub_motion_labels) == 3:
                                break
                        if len(sub_motion_labels) == 0:
                            sub_motion_labels.append("")
                        while len(sub_motion_labels) < 3:
                            sub_motion_labels.append(sub_motion_labels[-1])
                    else:
                        sub_motion_labels.append("")
                        sub_motion_labels.append("")
                        sub_motion_labels.append("")

                    motion_labels.append(sub_motion_labels)

                    motion_index += 1
        else:
            motion_files = [motion_file]
            motion_weights = [1.0]
            motion_timings = [[0, -1]]
            motion_fpses = [None]
            sub_motion_to_motion = [0]
            ref_respawn_offsets = [0]
            motion_labels = [["", "", ""]]
        return (
            motion_files,
            motion_weights,
            motion_timings,
            motion_fpses,
            sub_motion_to_motion,
            ref_respawn_offsets,
            motion_labels,
        )

    def _calc_frame_blend(self, time, len, num_frames, dt):
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = (time - frame_idx0 * dt) / dt

        return frame_idx0, frame_idx1, blend

    def _compute_motion_dof_vels(self, motion):
        # We pre-compute the dof vels in fk.
        return motion.dof_vels

    def fix_motion_heights(self, motion, skeleton_tree):
        body_heights = motion.global_translation[..., 2].clone()
        min_height = body_heights.min()

        motion.global_translation[..., 2] -= min_height
        return motion

    @staticmethod
    def _fix_motion_fps(motion, orig_fps, target_frame_rate, skeleton_tree):
        skip = int(np.round(orig_fps / target_frame_rate))

        downsampled_motion = {}
        for key in motion.keys():
            # if is torch.Tensor
            if isinstance(motion[key], torch.Tensor):
                downsampled_motion[key] = motion[key][::skip].clone()
            else:
                downsampled_motion[key] = copy.deepcopy(motion[key])

        downsampled_motion["fps"] = target_frame_rate
        return EasyDict(downsampled_motion)
    




@torch.jit.script
def quaternion_to_matrix(quaternions: torch.Tensor, w_last: bool) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions of shape (..., 4).
        w_last: If True, the real part of the quaternion is last.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if w_last:
        i, j, k, r = torch.unbind(quaternions, -1)
    else:
        r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

