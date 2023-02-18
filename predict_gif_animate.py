import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchmetrics
from matplotlib.animation import FuncAnimation, PillowWriter
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import config
from dataset import BratsDataset
from neural_network_v2 import UNet
from paths import Paths
from utils import DICE, RandomChoice


def getImageSlice(data, image_slice):
    return data[:, :, image_slice]


def getImageSlice_first_dim(data, image_slice):
    return data[image_slice, :, :]


if __name__ == "__main__":

    model_path = (
        "./output_models/model_17-2-2023-57_B2_E20_LR0.0001_WxH_64x64_64_128_256_512.pt"
    )
    model = UNet(
        in_channels=config.INPUT_CHANNELS,
        out_channels=config.OUTPUT_CHANNELS,
        # features=[32, 64, 128, 256],
        features=[64, 128, 256, 512],
    ).to(config.DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    paths = Paths(config.DATASET_PATH)
    image_paths, mask_paths = paths.get_file_path_list_multi_channel()
    # create datasets
    dataset = BratsDataset(
        image_paths=image_paths,  # [0:19],
        mask_paths=mask_paths,  # [0:19],
        # custom_transforms=transformations,
    )

    train_dataset, test_dataset = random_split(
        dataset,
        [1 - config.VAL_SPLIT, config.VAL_SPLIT],
        torch.Generator().manual_seed(42),
    )

    print(f"[INFO] {len(test_dataset)} test images found")

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        pin_memory=config.PIN_MEMORY,
        num_workers=os.cpu_count(),
    )

    prediction = None
    label = None
    image = None
    acc = torchmetrics.JaccardIndex(task="multiclass", num_classes=4).to(config.DEVICE)
    # score = torchmetrics.Dice(num_classes=4).to(config.DEVICE)
    score = DICE(num_classes=4).to(config.DEVICE)
    running_dice = 0.0
    print(f"{len(test_loader.dataset)}")
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move the inputs and labels to the specified device
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE).long()
            # Forward pass
            outputs = model(inputs)
            running_dice += score(outputs, labels)

            acc.update(outputs, labels)
            softmax = torch.softmax(outputs, dim=1)
            argmaxed = torch.argmax(softmax, 1)
            outputs = argmaxed

            prediction = outputs.cpu().float()
            prediction[prediction == 0] = torch.nan

            label = labels.cpu().float()
            label[label == 0] = torch.nan
            image = inputs.cpu()

    print(f"DICE: {running_dice/len(test_loader.dataset)}")
    print(f"Accuracy: {acc.compute()*100:.2f}%")
    print("Prediction shape:", prediction.shape)
    print("Label shape:", label.shape)
    print("Image shape:", image.shape)

    fig, ax = plt.subplots(ncols=2)

    def animate(index):
        ax[0].clear()
        ax[1].clear()
        data = getImageSlice_first_dim(prediction[0], index)
        input = getImageSlice_first_dim(image[0][0], index)
        label_img = getImageSlice_first_dim(label[0], index)

        ax[0].imshow(input, cmap="gray")
        ax[0].imshow(data, alpha=0.5, cmap="viridis")
        ax[0].set_title("Preditcion")

        ax[1].imshow(input, cmap="gray")
        ax[1].imshow(label_img, alpha=0.5, cmap="viridis")
        ax[1].set_title("Original")

    ani = FuncAnimation(fig, animate, frames=label[0].shape[0], interval=50)
    writer = PillowWriter(fps=12)
    ani.save("output_128.gif", writer=writer)

    plt.show()
