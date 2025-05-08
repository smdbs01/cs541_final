import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter


def main(history_path: str):
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"History file not found: {history_path}")

    history = np.load(history_path)

    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    val_acc = history["val_acc"]

    train_loss_x = np.arange(1, len(train_loss) + 1)
    val_loss_x = np.arange(10, len(val_loss) * 10 + 1, 10)
    val_acc_x = np.arange(10, len(val_acc) * 10 + 1, 10)

    train_loss_smooth = savgol_filter(train_loss, 15, 3) 

    def smooth_curve(x, y, num=300):
        spl = make_interp_spline(x, y, k=3)
        x_smooth = np.linspace(x.min(), x.max(), num)
        return x_smooth, spl(x_smooth)

    # Smooth the validation loss and accuracy
    val_loss_x_smooth, val_loss_smooth = smooth_curve(val_loss_x, val_loss)
    val_acc_x_smooth, val_acc_smooth = smooth_curve(val_acc_x, val_acc)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(
        train_loss_x,
        train_loss_smooth,
        "b-",
        alpha=0.7,
        linewidth=2,
        label="Train Loss",
    )

    ax1.plot(
        val_loss_x_smooth,
        val_loss_smooth,
        "r-",
        alpha=0.7,
        linewidth=2,
        label="Val Loss",
    )

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", color="b", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="b")

    ax2 = ax1.twinx()
    ax2.plot(
        val_acc_x_smooth,
        val_acc_smooth,
        "g-",
        alpha=0.7,
        linewidth=2,
        label="Val Acc",
    )
    ax2.set_ylabel("Accuracy", color="g", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="g")

    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="right", fontsize=10)

    ax1.grid(True, linestyle="--", alpha=0.6)
    plt.title("Training History", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig("smooth_training_history.png", dpi=300)


if __name__ == "__main__":
    main("checkpoints/history.npz")
