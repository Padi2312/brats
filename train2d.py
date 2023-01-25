import gc
import os
import time

import torch
import torch.cuda
import torchvision.transforms as transforms
from torch import save
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import config
from dataset2d import BratsDataset2D
from model2d import Unet
from paths import Paths

if __name__ == "__main__":
    print(
        f"[DEVICE] Use {config.DEVICE}({torch.cuda.get_device_name(torch.cuda.current_device())}) for training."
    )
    gc.collect()
    torch.cuda.empty_cache() if config.DEVICE == "cuda" else None

    paths = Paths(config.DATASET_PATH)
    image_paths, mask_paths = paths.get_file_path_list_multi_channel()

    # set transformations for data
    transformations = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(90),
        ]
    )

    # create datasets
    dataset = BratsDataset2D(
        image_paths=image_paths,
        mask_paths=mask_paths,
        custom_transforms=transformations,
    )

    train_dataset, test_dataset = random_split(
        dataset, [0.8, 0.2], torch.Generator().manual_seed(42)
    )

    print(f"[INFO] {len(dataset)} training images and {len(dataset)} test images found")
    print(
        f"[INFO] Len train_dataset: {len(train_dataset)} | Len test_dataset: {len(test_dataset)}"
    )

    # create train and test dataloaders
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.BATCH_SIZE,
        pin_memory=config.PIN_MEMORY,
        num_workers=os.cpu_count(),
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config.BATCH_SIZE,
        pin_memory=config.PIN_MEMORY,
        num_workers=os.cpu_count(),
    )

    # init model
    model = Unet(
        in_channels=config.INPUT_CHANNELS, out_channels=config.OUTPUT_CHANNELS
    ).to(config.DEVICE)
    # summary(unet, (4, 128, 128, 128))
    # Set model to train mode
    optimizer = SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=0.9)
    criterion = CrossEntropyLoss()

    print(f"[INFO] Start training network...")
    start_time = time.time()

    model.train()

    for epoch in range(config.NUM_EPOCHS):
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, position=0, leave=True):
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print the average loss for the epoch
        print(
            "Epoch {} Loss: {:.4f}".format(epoch + 1, running_loss / len(train_loader))
        )
        # Set the model to evaluation mode
        model.eval()
        # Initialize the running loss for the validation set
        val_loss = 0.0
        # Loop over the validation data

        with torch.no_grad():
            for inputs, labels in test_loader:
                # Move the inputs and labels to the specified device
                inputs = inputs.to(config.DEVICE)
                labels = labels.to(config.DEVICE)
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                # Update the running loss for the validation set
                val_loss += loss.item()

        print(
            "Epoch {} Val Loss: {:.4f}".format(epoch + 1, val_loss / len(test_loader))
        )

    save(model.state_dict(), config.MODEL_PATH)
