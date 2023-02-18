import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import config
from dataset import BratsDataset
from paths import Paths
import os


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool3d(2, 2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, features):
        super(UNet, self).__init__()
        # features = [32, 64, 128, 256]
        # features = [64, 128, 256, 512]
        features[:] = [x // 2 for x in features]

        self.inc = InConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        self.down4 = Down(features[3], features[3])

        self.up1 = Up(features[3], features[3], features[2])
        self.up2 = Up(features[2], features[2], features[1])
        self.up3 = Up(features[1], features[1], features[0])
        self.up4 = Up(features[0], features[0], features[0])
        self.outc = OutConv(features[0], out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


if __name__ == "__main__":

    paths = Paths(config.DATASET_PATH)
    image_paths, mask_paths = paths.get_file_path_list_multi_channel()
    image_paths, mask_paths = image_paths[:1], mask_paths[:1]

    train_ds = BratsDataset(image_paths=image_paths, mask_paths=mask_paths)
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        batch_size=config.BATCH_SIZE,
        pin_memory=config.PIN_MEMORY,
        num_workers=os.cpu_count(),
    )

    model = UNet(
        in_channels=config.INPUT_CHANNELS,
        out_channels=config.OUTPUT_CHANNELS,
        features=[16, 32, 64, 96],
    ).to(config.DEVICE, non_blocking=True)
    model.load_state_dict(torch.load("output_models/unet_model_best.pt"))
    model.eval()
    with torch.no_grad():
        for img, label in train_loader:
            img = img.to(config.DEVICE, non_blocking=True)
            label = label.to(config.DEVICE, non_blocking=True)
            output = model(img)
            output = torch.softmax(output, 1)
            argmax = torch.argmax(output, 1)
            print("Output", output.shape, torch.unique(output))
            print("Label", label.shape, torch.unique(label))
            print("Argmax", argmax.shape, torch.unique(argmax))
