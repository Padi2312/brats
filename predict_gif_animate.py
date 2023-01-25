from matplotlib.animation import FuncAnimation
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

import config
from dataset import BratsDataset
from model import UNet3D
from paths import Paths


def getImageSlice(data, image_slice):
    return data[:, :, image_slice]


def getImageSlice_first_dim(data, image_slice):
    return data[image_slice, :, :]


if __name__ == "__main__":

    model_path = "./output_models/unet_model50232.pt"
    # loaded = torch.load(model_path)
    # print(loaded["epoch"], loaded["loss"])
    # model = UNet(config.INPUT_CHANNELS,
    #              config.OUTPUT_CHANNELS).to(config.DEVICE)
    # model.load_state_dict(loaded['model_state_dict'])
    model = UNet3D(
        in_channels=config.INPUT_CHANNELS,
        out_channels=config.OUTPUT_CHANNELS,
        n_features=[32, 64, 128, 256],
    ).to(config.DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    paths = Paths(config.DATASET_PATH)
    image_paths, mask_paths = paths.get_file_path_list_multi_channel()

    # create datasets
    dataset = BratsDataset(
        image_paths=image_paths[:9],
        mask_paths=mask_paths[:9],
    )

    train_dataset, test_dataset = random_split(
        dataset, [0.8, 0.2], torch.Generator().manual_seed(42)
    )

    print(f"[INFO] {len(dataset)} training images and {len(dataset)} test images found")
    print(
        f"[INFO] Len train_dataset: {len(train_dataset)} | Len test_dataset: {len(test_dataset)}"
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        pin_memory=config.PIN_MEMORY,
        num_workers=os.cpu_count(),
    )

    predictions = []
    true_labels = []
    input_images = []
    amax = None
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move the inputs and labels to the specified device
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            # Forward pass
            outputs = model(inputs)
            input_images.append(inputs.cpu().numpy())

            # outputs[outputs == 0] = torch.nan
            print(f"{outputs.shape}")
            softmax = torch.softmax(outputs, dim=1)
            argmax = torch.where(outputs > 0.7, outputs, 1)
            amax = argmax.cpu().numpy()
            print(f"S {softmax.shape}")
            print(torch.max(softmax), softmax.shape)
            print(f"A {argmax.shape}")
            print(torch.max(argmax), argmax.shape)
            predicted = softmax.cpu().numpy()
            labels = labels.cpu().numpy()

            # Append the predictions and true labels to the lists
            predictions.append(predicted)
            true_labels.append(labels)

            # Concatenate the lists of predictions and true labels
        predictions = np.concatenate(predictions)
        true_labels = np.concatenate(true_labels)

    print("Prediction shape:", predictions[0].shape)
    print("Label shape:", true_labels[0].shape)
    print("Input image shape:", input_images[0][0].shape)

    torch.save(
        {"image": input_images, "label": true_labels, "prediction": predictions},
        "prediction_output.pt",
    )

    fig, ax = plt.subplots()

    def animate(index):
        ax.clear()
        data = getImageSlice_first_dim(predictions[0][0], index)
        data_1 = getImageSlice_first_dim(predictions[0][1], index)
        data_2 = getImageSlice_first_dim(predictions[0][2], index)
        data_3 = getImageSlice_first_dim(predictions[0][3], index)

        input = getImageSlice_first_dim(input_images[0][0][0], index)
        label = getImageSlice_first_dim(true_labels[0][0], index)

        #am = getImageSlice_first_dim(amax[0][3], index)

        ax.imshow(input, cmap="gray")
        ax.imshow(label, alpha=0.3, cmap="gray")
        #ax.imshow(am, alpha=0.9, cmap="Blues")
        # ax.imshow(data, alpha=0.25, cmap='Blues')
        # ax.imshow(data_1, alpha=0.25, cmap='Greens')
        ax.imshow(data_2, alpha=0.5, cmap="Reds")
        ax.imshow(data_3, alpha=0.5, cmap="Purples")

    ani = FuncAnimation(fig, animate, frames=predictions[0].shape[1], interval=50)
    plt.show()
