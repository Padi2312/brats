import gc
import os
import time

import torch
import torch.cuda
import torchmetrics
import torchvision.transforms as transforms
from torch import save
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import config
from dataset import BratsDataset
from neural_network_v2 import UNet
from paths import Paths
import utils

if __name__ == "__main__":
    print(
        f"[DEVICE] Use {config.DEVICE}({torch.cuda.get_device_name(torch.cuda.current_device())}) for training."
    )
    gc.collect()
    torch.cuda.empty_cache() if config.DEVICE == "cuda" else None

    paths = Paths(config.DATASET_PATH)
    image_paths, mask_paths = paths.get_file_path_list_multi_channel()

    transformations = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
        ]
    )

    # create datasets
    dataset = BratsDataset(
        image_paths=image_paths,
        mask_paths=mask_paths,
        custom_transforms=transformations,
    )

    

    train_dataset, test_dataset = random_split(
        dataset,
        [1 - config.VAL_SPLIT, config.VAL_SPLIT],
        torch.Generator().manual_seed(42),
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
    model = UNet(
        in_channels=config.INPUT_CHANNELS,
        out_channels=config.OUTPUT_CHANNELS,
        features=[64, 128, 256, 512],
    ).to(config.DEVICE)

    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    # criterion = CrossEntropyLoss()
    criterion = utils.DiceLoss().to(config.DEVICE)
    score = utils.DICE().to(config.DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    print(f"[INFO] Start training network...")
    start_time = time.time()
    model.train()
    # acc = torchmetrics.Accuracy(task="multilabel", num_labels=4).to(config.DEVICE)
    for epoch in range(config.NUM_EPOCHS):
        running_loss = 0.0
        dice_score = 0.0
        progress_bar = tqdm(train_loader, unit="batches")
        progress_bar.set_description(f"[Epoch {epoch+1}|Train]")
        for (inputs, labels) in progress_bar:
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # dice_score += score(outputs, labels).item()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{(loss.item()):.4f}")
        print(
            f"[Epoch {epoch + 1}|Result] Loss: {running_loss / len(train_loader):.4f}"
        )

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move the inputs and labels to the specified device
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    print(f"Epoch {epoch + 1} Val Loss: {val_loss / len(test_loader):.4f}")

    save(model.state_dict(), config.MODEL_PATH)
