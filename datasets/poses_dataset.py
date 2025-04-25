import json
import math
import os
import random

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class PoseDataset(Dataset):
    def __init__(
        self,
        split_file: str,
        split: list[str],
        pose_root: str,
        sample_strategy="rnd_start",
        num_samples=50,
        num_copies=4,
    ):
        assert os.path.exists(split_file), f"Index file not found: {split_file}"
        assert os.path.exists(pose_root), f"Pose directory not found: {pose_root}"

        self.data = []
        self.label_encoder = LabelEncoder()
        self.pose_root = pose_root
        self.sample_strategy = sample_strategy
        self.num_samples = num_samples
        self.num_copies = num_copies

        # Initialize dataset
        self._make_dataset(split_file, split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        video_id, gloss_cat, frame_start, frame_end = self.data[index]
        x = self._load_poses(video_id, frame_start, frame_end)
        y = torch.tensor(gloss_cat, dtype=torch.long)
        return x, y, video_id

    def _make_dataset(self, index_file_path, split):
        with open(index_file_path, "r") as f:
            content = json.load(f)  # Read split file Json

        # Get all unique classes
        classes = set()
        for vid in content.values():
            classes.add(vid["action"][0])
        self.label_encoder.fit(sorted(classes))

        # Parse dataset
        for vid, metadata in content.items():
            # Filter split
            if metadata["subset"] not in split:
                continue

            # Get label
            class_id = metadata["action"][0]
            gloss_cat = self.label_encoder.transform([class_id])[0]  # type: ignore

            # Handle video ID
            if len(vid) == 5:
                frame_start = 0
                total_frames = metadata["action"][2] - metadata["action"][1]
            elif len(vid) == 6:
                frame_start = metadata["action"][1]
                total_frames = metadata["action"][2] - metadata["action"][1]

            self.data.append(
                (
                    vid,
                    gloss_cat,
                    frame_start,  # type: ignore
                    frame_start + total_frames - 1,  # type: ignore
                )
            )

    def _load_poses(self, video_id, frame_start, frame_end):
        """Load and preprocess pose data for a given video ID"""
        npz_path = os.path.join(self.pose_root, f"{video_id}.npz")
        data = np.load(npz_path)

        # for part in ["pose", "left_hand", "right_hand"]:
        #     if np.all(data[part][..., :2] == 0):
        #         print(f"Warning: Zero-values in {part} for {video_id}")

        # Get each part's data and transpose dimensions (T, N, 3) -> (N, T, 3)
        face = data["face"].transpose(1, 0, 2)  # (478, T, 3)
        pose = data["pose"].transpose(1, 0, 2)  # (33, T, 3)
        lhand = data["left_hand"].transpose(1, 0, 2)  # (21, T, 3)
        rhand = data["right_hand"].transpose(1, 0, 2)  # (21, T, 3)

        # Merge all keypoints (553, T, 3)
        full_pose = np.concatenate([face, pose, lhand, rhand], axis=0)

        # Sample frame indices
        frames_to_sample = self._get_sample_indices(
            frame_start, frame_end, total_frames=full_pose.shape[1]
        )

        # Extract and preprocess data
        sampled = self._process_frames(full_pose, frames_to_sample)
        return torch.FloatTensor(sampled)

    def _get_sample_indices(self, orig_start, orig_end, total_frames):
        """Generate frame indices based on sampling strategy"""
        # Convert to 0-based indices
        frame_start = max(0, orig_start)
        frame_end = min(total_frames - 1, orig_end)
        frame_end = max(frame_start, frame_end)  # Ensure valid range

        # Add debug print
        # print(
        #     f"Sampling: video_frames={total_frames}, req_start={orig_start}, req_end={orig_end}, adj_start={frame_start}, adj_end={frame_end}"
        # )

        if self.sample_strategy == "rnd_start":
            return rand_start_sampling(frame_start, frame_end, self.num_samples)
        if self.sample_strategy == "seq":
            return sequential_sampling(frame_start, frame_end, self.num_samples)
        if self.sample_strategy == "k_copies":
            return k_copies_sampling(
                frame_start, frame_end, self.num_samples, self.num_copies
            )
        raise ValueError(f"Unknown sampling strategy: {self.sample_strategy}")

    def _process_frames(self, pose_data, indices):
        """Preprocessing pipeline for sampled frames"""
        # Select required frames
        sampled = pose_data[:, indices, :2]  # Only take x,y coordinates

        # Handle padding if needed
        if sampled.shape[1] < self.num_samples:
            padding = np.repeat(
                sampled[:, [-1], :], self.num_samples - sampled.shape[1], axis=1
            )
            sampled = np.concatenate([sampled, padding], axis=1)

        return sampled.transpose(1, 0, 2)  # (T, Joint, 2)


# Sampling strategy functions remain unchanged
def rand_start_sampling(start, end, num_samples):
    """Improved random start sampling function"""
    available_frames = end - start + 1

    # Handle invalid ranges
    if available_frames <= 0:
        return [start] * num_samples  # Return padding values

    if available_frames >= num_samples:
        sample_start = random.randint(start, end - num_samples + 1)  # Corrected range
        return list(range(sample_start, sample_start + num_samples))

    # When insufficient frames: full sampling + end padding
    valid_frames = list(range(start, end + 1))
    return valid_frames + [valid_frames[-1]] * (num_samples - available_frames)


def sequential_sampling(start, end, num_samples):
    indices = np.linspace(start, end, num=num_samples, dtype=int)
    return indices.tolist()


def k_copies_sampling(start, end, num_samples, num_copies):
    total_frames = end - start + 1
    if total_frames <= num_samples:
        return list(range(start, end + 1)) * num_copies

    stride = max((total_frames - num_samples) // (num_copies - 1), 1)
    samples = []
    for i in range(num_copies):
        sample_start = start + i * stride
        sample_end = min(sample_start + num_samples - 1, end)
        if sample_end - sample_start + 1 < num_samples:
            sample_start = max(sample_end - num_samples + 1, start)
        samples.extend(range(sample_start, sample_end + 1))
    return samples[: num_samples * num_copies]
