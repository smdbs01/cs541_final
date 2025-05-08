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
from tqdm import tqdm

import videotransforms
from config import TrainConfig
from datasets.poses_dataset import PoseDataset
from datasets.vision_dataset import NSLT as VisionDataset
from models.cnn_lstm import CNNLSTM
from models.pose_1dcnn import GestureModel, PosePreprocess


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def test(configs: TrainConfig):
    root_dir = configs.data_root
    train_split_file = os.path.join(root_dir, configs.train_split)

    test_dataset = PoseDataset(
        split_file=train_split_file,
        split=["test"],
        pose_root=configs.pose_path,
        sample_strategy="rnd_start",
        num_samples=configs.n_samples,
        num_copies=configs.num_copies,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=configs.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    LHAND = np.arange(478 + 33, 478 + 33 + 21).tolist()
    RHAND = np.arange(478 + 33 + 21, 478 + 33 + 21 + 21).tolist()

    POINT_LANDMARKS = LHAND + RHAND

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocess = PosePreprocess(POINT_LANDMARKS).to(device)
    model = GestureModel(configs.n_samples, 252, 512, 100, 0).to(device)

    weights_path = "checkpoints/model_h512_b128.pth"
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        model.load_state_dict(torch.load(weights_path))
    else:
        print(f"Weights file not found: {weights_path}")
        return

    total = 0
    top1 = 0
    top5 = 0
    top10 = 0
    for i, (x, y, _) in enumerate(tqdm(test_dataloader)):
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            x = preprocess(x)
            output = model(x)
            results = torch.sort(output, dim=1, descending=True)
            is_top1 = results.indices[:, 0] == y
            is_top5 = results.indices[:, :5] == y.unsqueeze(1)
            is_top10 = results.indices[:, :10] == y.unsqueeze(1)
            top1 += is_top1.sum().item()
            top5 += is_top5.sum().item()
            top10 += is_top10.sum().item()
            total += y.size(0)
    print(f"Top-1 Accuracy: {top1 / total:.4f}")
    print(f"Top-5 Accuracy: {top5 / total:.4f}")
    print(f"Top-10 Accuracy: {top10 / total:.4f}")


if __name__ == "__main__":
    set_seed(42)

    # WLASL setting
    config_file = "configs/default.yaml"

    configs = TrainConfig(config_file)

    test(configs)
