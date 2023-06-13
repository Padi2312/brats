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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
import utils.metrics as metrics
from dataset import get_train_val_test_dataset, load_prepared_img_paths
from model import UNet
from utils.losses import DiceCE, DiceLossV4


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

    def __get_model_name(self, info=""):
        num_data = (
            len(self.train_dataset) + len(self.val_dataset) + len(self.test_dataset)
        )
        batch_size = self.batch_size
        learning_rate = str(self.optimizer.param_groups[0]["lr"]).replace(".", "")
        if len(info) > 0:
            return f"{num_data}_lr{learning_rate}_b{batch_size}_{info}"
        else:
            return f"{num_data}_lr{learning_rate}_b{batch_size}"

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

    def __save_model(self, save_path: str):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            save_path,
        )

    def evaluate(self, model_name: str, mean=True):
        loaded_data = torch.load(
            os.path.join(self.save_folder, model_name, model_name + ".pt")
        )
        self.model.load_state_dict(loaded_data["model_state_dict"])
        self.model.eval()
        dice = metrics.DiceScore(activation=None, mean=mean)
        jaccard = metrics.JaccardScore(activation=None, mean=mean)
        pixel_accuracy = metrics.PixelAccuracy(mean=mean)
        if mean:
            dice_score = 0.0
            jaccard_score = 0.0
            pixel_accuracy_score = 0.0
        else:
            dice_score = torch.zeros(4).to(self.device)

        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc=f"Test ({model_name})"):
                # Move the inputs and labels to the specified device
                inputs = inputs.to(config.DEVICE)
                labels = labels.to(config.DEVICE).long()
                # Forward pass
                outputs = model(inputs)
                softmax = torch.softmax(outputs, dim=1)
                argmaxed = torch.argmax(softmax, 1)
                argmaxed = torch.nn.functional.one_hot(argmaxed, 4).movedim(-1, 1)

                dice_score += dice(argmaxed, labels).to(self.device)
                jaccard_score += jaccard(argmaxed, labels).to(self.device)
                pixel_accuracy_score += pixel_accuracy(argmaxed, labels)

        dice_score /= len(self.test_dataset)
        jaccard_score /= len(self.test_dataset)
        pixel_accuracy_score /= len(self.test_dataset)

        if mean:
            print(f"[{model_name}] Test DICE score: {dice_score*100:.2f}")
            print(f"[{model_name}] Test Jaccard score: {jaccard_score*100:.2f}")
            print(f"[{model_name}] Pixel Accuracy: {pixel_accuracy_score*100:.2f}")
        else:
            print(f"[{model_name}] Test DICE score:", dice_score.cpu())

        file = open(
            os.path.join(self.save_folder, model_name, model_name + "_result.txt"), "w"
        )
        text = f"Dice Score: {dice_score}\nJaccard Score: {jaccard_score}\nPixel Accuracy: {pixel_accuracy_score}"
        file.write(text)
        file.close()
        return dice_score

    def evaluate_image(self, model_name: str, img_label: str):
        loaded_data = torch.load(
            os.path.join(self.save_folder, model_name, model_name + ".pt")
        )
        self.model.load_state_dict(loaded_data["model_state_dict"])
        self.model.eval()
        dice = metrics.DiceScore(activation=None)
        jaccard = metrics.JaccardScore(activation=None)

        dice_score = 0.0
        jaccard_score = 0.0
        img = None
        outputs = None
        with torch.no_grad():
            inputs = img_label[0].to(config.DEVICE)
            labels = img_label[1].to(config.DEVICE).long()
            # Forward pass
            outputs = model(inputs)
            softmax = torch.softmax(outputs, dim=1)
            argmaxed = torch.argmax(softmax, 1)

            # Set output image
            outputs = argmaxed
            img = inputs

            argmaxed = torch.nn.functional.one_hot(argmaxed, 4).movedim(-1, 1)

            dice_score += dice(argmaxed, labels).to(self.device)
            jaccard_score += jaccard(argmaxed, labels).to(self.device)

        print(f"[{model_name}] Test DICE score: {dice_score*100:.2f}")
        print(f"[{model_name}] Test Jaccard score: {jaccard_score*100:.2f}")

        # Prepare for show
        outputs = outputs.float()
        outputs[outputs == 0] = torch.nan
        outputs = outputs.cpu()

        img = img.float()
        img = img.cpu()

        labels = torch.argmax(labels, dim=1)
        labels = labels.float()
        labels[labels == 0] = torch.nan
        labels = labels.cpu()

        _, ax = plt.subplots(ncols=2)
        ax[0].imshow(img[0][0], cmap="gray")
        ax[0].imshow(outputs[0], alpha=0.4, cmap="viridis")
        ax[0].set_title("Prediction")

        ax[1].imshow(img[0][0], cmap="gray")
        ax[1].imshow(labels[0], alpha=0.4, cmap="viridis")
        ax[1].set_title("Ground truth")
        plt.show()
        return dice_score

    def show_test_samples(self, model_path: str, amount=128):
        loaded_data = torch.load(
            os.path.join(self.save_folder, model_path, model_path + ".pt")
        )
        self.model.load_state_dict(loaded_data["model_state_dict"])
        self.model.eval()

        score = metrics.DiceScore(activation=None)
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
                current_score = score(outputs, labels)
                dice_score += current_score
                if len(torch.unique(outputs)) < 2:
                    continue

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
                print("Dice:", current_score)
                if count == amount:
                    break

        dice_score /= count
        print(f"Test DICE score: {dice_score*100:.2f}")
        return dice_score

    def train(self, n_epochs, log_interval=1, model_info: str = ""):
        print(
            f"[DEVICE] Use {self.device}({torch.cuda.get_device_name(torch.cuda.current_device())}) for training."
        )
        print(
            f"[INFO] Len train_dataset: {len(self.train_dataset)} | Len val_dataset: {len(self.val_dataset)} | Len test_dataset: {len(self.test_dataset)}"
        )
        try:
            model_name = self.__get_model_name(info=model_info)
            folder = os.path.join(self.save_folder, model_name)

            self.writer = SummaryWriter(folder)
            model_save_path = os.path.join(folder, model_name + ".pt")
            best_val_loss = np.inf

            for epoch in range(1, n_epochs + 1):
                start_time = time.time()

                train_loss = self.__train_epoch()
                val_loss = self.__validate_epoch()

                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)

                self.writer.add_scalar("Loss/Train", train_loss, epoch)
                self.writer.add_scalar("Loss/Val", val_loss, epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.__save_model(model_save_path)
                    print(
                        f"Saving the model at epoch {epoch} with validation loss {val_loss:.4f}"
                    )

                if epoch % log_interval == 0:
                    elapsed_time = time.time() - start_time
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"Epoch {epoch}/{n_epochs} | LR: {current_lr}| Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Time: {elapsed_time:.2f}s"
                    )
                if self.lr_scheduler != None:
                    self.lr_scheduler.step()
            self.writer.close()
        except KeyboardInterrupt:  # type: ignore
            sys.exit(0)
            pass


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    data = load_prepared_img_paths(config.DATASET_PROCESSED_PATH_128_CC)
    data = data[0:25000]
    transformations = None
    # get datasets
    train_dataset, val_dataset, test_dataset = get_train_val_test_dataset(
        data=data,
        transforms=transformations,
    )
    model = UNet(
        in_channels=config.INPUT_CHANNELS,
        out_channels=config.OUTPUT_CHANNELS,
        features=[64, 128, 256, 512],
    )
    optimizer = Adam(model.parameters(), lr=0.001)  # 0.0001 # 0.01
    lr_scheduler = StepLR(optimizer=optimizer, step_size=5, gamma=0.1)
    criterion = DiceCE()

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
    model_name_default = "25000_lr001_b32"
    model_name_b8 = "25000_lr001_b8"
    model_name_b64 = "25000_lr001_b64"
    model_name_lr01 = "25000_lr01_b32"
    model_name_lr0001 = "25000_lr0001_b32"
    model_name_nocc = "25000_lr001_b32_nocc"


    #trainer.show_test_samples("25000_lr0001_b32_perf")
    # trainer.evaluate(model_name_b8)
    # trainer.evaluate(model_name_b64)
    # trainer.evaluate(model_name_lr01)
    # trainer.evaluate(model_name_lr0001)
    # trainer.evaluate(model_name_nocc)
    # trainer.show_test_samples(model_name)
# %%
