import argparse
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torchvision import transforms

import videotransforms
from config import TrainConfig
from datasets.poses_dataset import PoseDataset
from datasets.vision_dataset import NSLT as VisionDataset


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def train_vision(configs: TrainConfig, save_model: str, weights: Optional[str] = None):
    root_dir = configs.data_root
    train_split_file = os.path.join(root_dir, configs.train_split)

    # setup dataset
    train_transforms = transforms.Compose(
        [
            videotransforms.RandomCrop(224),
            videotransforms.RandomHorizontalFlip(),
        ]
    )
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = VisionDataset(
        train_split_file, ["train"], root_dir, "rgb", train_transforms
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=configs.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_dataset = VisionDataset(
        train_split_file, ["val"], root_dir, "rgb", test_transforms
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=configs.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
    )

    dataloaders = {"train": dataloader, "val": val_dataloader}
    datasets = {"train": dataset, "val": val_dataset}

    # Get a batch of training data
    for inputs, labels, vid_id in dataloaders["train"]:
        print(f"Batch inputs shape: {inputs.shape}")
        print(f"Batch labels shape: {labels.shape}")
        print(f"Video ID: {vid_id}")
        break


def train_pose(configs: TrainConfig, save_model: str, weights: Optional[str] = None):
    root_dir = configs.data_root
    train_split_file = os.path.join(root_dir, configs.train_split)

    dataset = PoseDataset(
        split_file=train_split_file,
        split=["train"],
        pose_root=configs.pose_path,
        sample_strategy="rnd_start",
        num_samples=configs.n_samples,
        num_copies=configs.num_copies,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=configs.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_dataset = PoseDataset(
        split_file=train_split_file,
        split=["val"],
        pose_root=configs.pose_path,
        sample_strategy="k_copies",
        num_samples=configs.n_samples,
        num_copies=configs.num_copies,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=configs.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
    )

    dataloaders = {"train": dataloader, "val": val_dataloader}
    datasets = {"train": dataset, "val": val_dataset}

    print(f"Number of training samples: {len(datasets['train'])}")
    print(f"Number of validation samples: {len(datasets['val'])}")


def run(
    configs: TrainConfig,
    save_model="",
    weights=None,
):
    print(configs)
    root_dir = configs.data_root
    if not os.path.exists(root_dir):
        raise ValueError("Data root directory does not exist: {}".format(root_dir))

    train_split_file = os.path.join(root_dir, configs.train_split)
    if not os.path.exists(train_split_file):
        raise ValueError("Train split file does not exist: {}".format(train_split_file))

    mode = configs.mode
    if mode not in ["vision", "pose"]:
        raise ValueError("Mode must be either 'vision' or 'pose', got {}".format(mode))

    if mode == "vision":
        train_vision(
            configs=configs,
            save_model=save_model,
            weights=weights,
        )
    else:
        train_pose(
            configs=configs,
            save_model=save_model,
            weights=weights,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-save_model", type=str, default="checkpoints/")
    parser.add_argument("-weights", type=str)

    args = parser.parse_args()

    # WLASL setting
    config_file = "configs/default.yaml"

    configs = TrainConfig(config_file)

    save_models, weights = args.save_model, args.weights

    run(
        configs=configs,
        save_model="checkpoints/",
        weights=None,
    )
