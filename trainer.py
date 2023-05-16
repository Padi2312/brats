# %% matplotlib inline
import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.cuda
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, StepLR, _LRScheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import config
import utils.metrics as metrics
from dataset import get_train_val_test_dataset, load_prepared_img_paths
from model import UNet
from utils.losses import DiceCE


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        save_folder: str,
        lr_scheduler: _LRScheduler = None,
        batch_size=2,
        device="cuda",
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.save_folder = save_folder

        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)

        self.history = {"train_loss": [], "val_loss": []}
        # self.writer = SummaryWriter(log_dir=save_folder)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
            pin_memory=config.PIN_MEMORY,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=config.PIN_MEMORY,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=config.PIN_MEMORY,
        )

        self.model.to(self.device)

    def __train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for inputs, targets in tqdm(self.train_loader, desc="Training"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def __validate_epoch(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validation"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()

        return running_loss / len(self.val_loader)

    def __save_model(self, epoch: int, loss: int, model_save_name: str):
        model_path = os.path.join(self.save_folder, f"{model_save_name}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss,
            },
            model_path,
        )

    def plot_losses(self, info=None):
        matplotlib.use("Agg")
        x_axis_values = range(len(self.history["train_loss"]))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x_axis_values, self.history["train_loss"], label="Train Loss")
        ax.plot(x_axis_values, self.history["val_loss"], label="Validation Loss")
        ax.set_xticks(x_axis_values)
        ax.set(xlabel="Epochs", ylabel="Loss", title="Training and Validation Losses")
        ax.legend()
        path = info if f"history-{info}.png" != None else "./history.png"
        fig.savefig(os.path.join(config.MODEL_FOLDER_PATH, path))
        plt.close(fig)

    def evaluate(self, model_path: str, mean=True):
        loaded_data = torch.load(os.path.join(self.save_folder, model_path))
        self.model.load_state_dict(loaded_data["model_state_dict"])
        self.model.eval()

        score = metrics.DiceScore(activation=None, mean=mean)
        # score = metrics.dice_score_v2

        if mean:
            dice_score = 0.0
        else:
            dice_score = torch.zeros(4).to(self.device)

        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc="Test"):
                # Move the inputs and labels to the specified device
                inputs = inputs.to(config.DEVICE)
                labels = labels.to(config.DEVICE).long()
                # Forward pass
                outputs = model(inputs)
                softmax = torch.softmax(outputs, dim=1)
                argmaxed = torch.argmax(softmax, 1)
                argmaxed = torch.nn.functional.one_hot(argmaxed, 4).movedim(-1, 1)
                dice_score += score(argmaxed, labels).to(self.device)
        dice_score /= len(self.test_dataset)
        if mean:
            print(f"Test DICE score: {dice_score*100:.2f}")
        else:
            print(f"Test DICE score:", dice_score.cpu())

        return dice_score

    def show_test_samples(self, model_path: str, amount=128):
        loaded_data = torch.load(os.path.join(self.save_folder, model_path))
        self.model.load_state_dict(loaded_data["model_state_dict"])
        self.model.eval()

        score = metrics.DiceScore()
        dice_score = 0.0

        count = 0
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc="Test"):
                # Move the inputs and labels to the specified device
                inputs = inputs.to(config.DEVICE)
                labels = labels.to(config.DEVICE).long()
                # Forward pass
                net_output = model(inputs)

                # Process for show image
                outputs = torch.argmax(torch.softmax(net_output, dim=1), dim=1)
                if len(torch.unique(outputs)) < 2:
                    continue
                dice_score += score(net_output, labels)

                count += 1
                outputs = outputs.float()
                outputs[outputs == 0] = torch.nan

                outputs = outputs.cpu()
                labels = torch.argmax(labels, dim=1)
                labels = labels.float()
                labels[labels == 0] = torch.nan
                labels = labels.cpu()
                img = inputs.cpu()

                fig, ax = plt.subplots(ncols=2)
                ax[0].imshow(img[0][0], cmap="gray")
                ax[0].imshow(outputs[0], alpha=0.4, cmap="viridis")
                ax[0].set_title("Prediction")

                ax[1].imshow(img[0][0], cmap="gray")
                ax[1].imshow(labels[0], alpha=0.4, cmap="viridis")
                ax[1].set_title("Ground truth")
                plt.show()
                if count == amount:
                    break

        dice_score /= count
        print(f"Test DICE score: {dice_score*100:.2f}")
        return dice_score

    def train(self, n_epochs, log_interval=1, model_info: str = ""):
        try:
            best_val_loss = np.inf

            for epoch in range(1, n_epochs + 1):
                start_time = time.time()

                train_loss = self.__train_epoch()
                val_loss = self.__validate_epoch()

                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.__save_model(epoch, val_loss, model_info)
                    print(
                        f"Saving the model at epoch {epoch} with validation loss {val_loss:.4f}"
                    )

                if epoch % log_interval == 0:
                    elapsed_time = time.time() - start_time
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"Epoch {epoch}/{n_epochs} | LR: {current_lr}| Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Time: {elapsed_time:.2f}s"
                    )

                self.plot_losses(model_info.split(".")[0] + ".png")
                if self.lr_scheduler != None:
                    self.lr_scheduler.step()
        except KeyboardInterrupt:  # type: ignore
            sys.exit(0)
            pass


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    print(
        f"[DEVICE] Use {config.DEVICE}({torch.cuda.get_device_name(torch.cuda.current_device())}) for training."
    )
    data = load_prepared_img_paths(config.DATASET_PROCESSED_PATH)
    data = data[0:25000]

    transformations = transforms.Compose(
        [
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomAffine(degrees=90),
            # transforms.Resize(
            #     (config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
            #     transforms.InterpolationMode.NEAREST,
            # )
            # transforms.RandomRotation(90),
            transforms.Normalize(
                mean=[341.1601, 467.0914, 279.2286, 219.8044],
                std=[395.8610, 557.0898, 369.1520, 270.5645],
                inplace=True,
            )
        ]
    )

    # get datasets
    train_dataset, val_dataset, test_dataset = get_train_val_test_dataset(
        data=data,
        transforms=transformations,
    )
    print(
        f"[INFO] Len train_dataset: {len(train_dataset)} | Len val_dataset: {len(val_dataset)} | Len test_dataset: {len(test_dataset)}"
    )
    model = UNet(
        in_channels=config.INPUT_CHANNELS,
        out_channels=config.OUTPUT_CHANNELS,
        features=[64, 128, 256, 512],
        # features=[128, 256, 512, 1024],
    )
    # model = torch.compile(model)

    optimizer = Adam(model.parameters(), lr=0.01)  # 0.0001
    # lr_scheduler = ExponentialLR(optimizer, gamma=0.1)
    # lr_scheduler = StepLR(optimizer=optimizer, step_size=5, gamma=0.1)
    criterion = DiceCE()
    # criterion = torch.compile(criterion)

    trainer = Trainer(
        model=model,
        batch_size=32,
        optimizer=optimizer,
        criterion=criterion,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_dataset=train_dataset,
        save_folder=config.MODEL_FOLDER_PATH,
        # lr_scheduler=lr_scheduler,
    )
    model_name = "20000-001lr-128-x2.pt"

    trainer.train(15, model_info=model_name)
    # trainer.evaluate(model_name)
    # trainer.show_test_samples(model_name)

# %%
