import random

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn

import config


def load_nifti_data(path: str):
    img = sitk.ReadImage(path, sitk.sitkFloat32)
    return sitk.GetArrayFromImage(img)


def get_image_slice(data, image_slice: int):
    return data[image_slice, :, :]


class DiceLoss(nn.Module):
    def __init__(self, num_classes=4, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target = torch.nn.functional.one_hot(
            target.long(), num_classes=self.num_classes
        )
        target = target.movedim(-1, 1)
        intersection = torch.sum(pred * target, dim=(2, 3, 4))
        union = torch.sum(pred, dim=(2, 3, 4)) + torch.sum(target, dim=(2, 3, 4))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class DICE(nn.Module):
    def __init__(self, num_classes=4, smooth=1.0):
        super(DICE, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def __call__(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        #pred = torch.argmax(pred, dim=1)
        target = torch.nn.functional.one_hot(
            target.long(), num_classes=self.num_classes
        )
        target = target.movedim(-1, 1)
        intersection = torch.sum(pred * target, dim=(2, 3, 4))
        union = torch.sum(pred, dim=(2, 3, 4)) + torch.sum(target, dim=(2, 3, 4))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return dice.mean()


class RandomChoice(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t = random.choice(self.transforms)

    def __call__(self, img):
        return self.t(img)
