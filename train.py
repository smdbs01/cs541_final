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


def train_vision(configs: TrainConfig, save_model: str, weights: Optional[str] = None):
    root_dir = configs.data_root
    video_root = configs.video_path
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
        train_split_file, ["train"], video_root, "rgb", train_transforms
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=configs.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    val_dataset = VisionDataset(
        train_split_file, ["val"], video_root, "rgb", test_transforms
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=configs.batch_size,
        shuffle=True,
        pin_memory=False,
    )

    dataloaders = {"train": dataloader, "val": val_dataloader}
    datasets = {"train": dataset, "val": val_dataset}
    num_classes = len(dataset.classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNLSTM(3, 256, 2, num_classes).to(device)
    for param in model.resnet.parameters():
        param.requires_grad = False

    if weights is not None:
        print(f"Loading weights from {weights}")
        model.load_state_dict(torch.load(weights, map_location=device), strict=False)

    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(100):
        running_loss = 0.0
        total = 0
        correct = 0
        model.train()
        # use tdqm
        t = tqdm(
            dataloaders["train"],
            desc=f"Epoch {epoch + 1}/{100}",
            unit="batch",
            total=len(dataloaders["train"]),
        )
        for inputs, labels, vid_id in t:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # print(f"Epoch [{epoch + 1}/{100}], Loss: {running_loss / len(dataset):.4f}")
        t.write(f"Loss: {running_loss / len(dataset):.4f}")
        t.write(f"Accuracy: {100 * correct / total:.2f}%")
        # t.write(f"Learning rate: {scheduler.get_last_lr()[0]}")

        if (epoch + 1) % 10 == 0:
            # Validation accuracy
            with torch.no_grad():
                model.eval()
                val_loss = 0.0
                val_total = 0
                val_correct = 0
                for inputs, labels, vid_id in dataloaders["val"]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

                t.write(f"Validation Loss: {val_loss / len(val_dataset):.4f}")
                t.write(f"Validation Accuracy: {100 * val_correct / val_total:.2f}%")

            torch.save(
                model.state_dict(),
                os.path.join(save_model, f"model_epoch_{epoch + 1}.pth"),
            )
            print(f"Model saved at epoch {epoch + 1}")

    return model


def train_pose(configs: TrainConfig, save_model: str, weights: Optional[str] = None):
    os.makedirs(save_model, exist_ok=True)
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
        pin_memory=True,
    )

    val_dataset = PoseDataset(
        split_file=train_split_file,
        split=["val"],
        pose_root=configs.pose_path,
        sample_strategy="rnd_start",
        num_samples=configs.n_samples,
        num_copies=configs.num_copies,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=configs.batch_size,
        shuffle=True,
        pin_memory=False,
    )

    dataloaders = {"train": dataloader, "val": val_dataloader}
    datasets = {"train": dataset, "val": val_dataset}

    # 478, 33, 21, 21 = 553
    NOSE = [1, 2, 98, 327]
    LNOSE = [98]
    RNOSE = [327]
    LIP = [
        0,
        61,
        185,
        40,
        39,
        37,
        267,
        269,
        270,
        409,
        291,
        146,
        91,
        181,
        84,
        17,
        314,
        405,
        321,
        375,
        78,
        191,
        80,
        81,
        82,
        13,
        312,
        311,
        310,
        415,
        95,
        88,
        178,
        87,
        14,
        317,
        402,
        318,
        324,
        308,
    ]
    LLIP = [84, 181, 91, 146, 61, 185, 40, 39, 37, 87, 178, 88, 95, 78, 191, 80, 81, 82]
    RLIP = [
        314,
        405,
        321,
        375,
        291,
        409,
        270,
        269,
        267,
        317,
        402,
        318,
        324,
        308,
        415,
        310,
        311,
        312,
    ]

    POSE = [510, 512, 514, 511, 513, 515, 522, 523]
    LPOSE = [523, 515, 513, 511]
    RPOSE = [522, 514, 512, 510]

    REYE = [
        33,
        7,
        163,
        144,
        145,
        153,
        154,
        155,
        133,
        246,
        161,
        160,
        159,
        158,
        157,
        173,
    ]
    LEYE = [
        263,
        249,
        390,
        373,
        374,
        380,
        381,
        382,
        362,
        466,
        388,
        387,
        386,
        385,
        384,
        398,
    ]
    LHAND = np.arange(478 + 33, 478 + 33 + 21).tolist()
    RHAND = np.arange(478 + 33 + 21, 478 + 33 + 21 + 21).tolist()

    POINT_LANDMARKS = LHAND + RHAND

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocess = PosePreprocess(
        point_landmarks=POINT_LANDMARKS,
    ).to(device)
    model = GestureModel(
        max_len=configs.n_samples,
        in_channels=len(POINT_LANDMARKS) * 6,
        dim=256,
        num_classes=100,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    for epoch in range(100):
        running_loss = 0.0
        total = 0
        correct = 0
        model.train()
        # use tdqm
        t = tqdm(
            dataloaders["train"],
            desc=f"Epoch {epoch + 1}/{100}",
            unit="batch",
            total=len(dataloaders["train"]),
        )
        for inputs, labels, vid_id in t:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            inputs = preprocess(inputs)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # print(f"Epoch [{epoch + 1}/{100}], Loss: {running_loss / len(dataset):.4f}")
        scheduler.step()
        t.write(f"Loss: {running_loss / len(dataset):.4f}")
        # t.write(f"Accuracy: {100 * correct / total:.2f}%")
        # t.write(f"Learning rate: {scheduler.get_last_lr()[0]}")

        history["train_loss"].append(running_loss / len(dataset))

        if (epoch + 1) % 10 == 0:
            # Validation accuracy
            with torch.no_grad():
                model.eval()
                val_loss = 0.0
                val_total = 0
                val_correct = 0
                for inputs, labels, vid_id in dataloaders["val"]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    inputs = preprocess(inputs)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

                t.write(f"Validation Loss: {val_loss / len(val_dataset):.4f}")
                t.write(f"Validation Accuracy: {100 * val_correct / val_total:.2f}%")

                history["val_loss"].append(val_loss / len(val_dataset))
                history["val_acc"].append(val_correct / val_total)

    torch.save(
        model.state_dict(),
        os.path.join(save_model, f"model_h{model.dim}_b{configs.batch_size}.pth"),
    )

    np.savez_compressed("checkpoints/history.npz", **history, allow_pickle=True)

    return model, optimizer, scheduler, criterion


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
        print("Training vision model...")
        train_vision(
            configs=configs,
            save_model=save_model,
            weights=weights,
        )
    elif mode == "pose":
        print("Training pose model...")
        train_pose(
            configs=configs,
            save_model=save_model,
            weights=weights,
        )
    else:
        raise ValueError("Mode must be either 'vision' or 'pose', got {}".format(mode))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-save_model", type=str, default="checkpoints/")
    parser.add_argument("-weights", type=str)

    args = parser.parse_args()

    # WLASL setting
    config_file = "configs/default.yaml"

    configs = TrainConfig(config_file)

    save_models, weights = args.save_model, args.weights

    set_seed(42)

    run(
        configs=configs,
        save_model="checkpoints/",
        weights=None,
    )
