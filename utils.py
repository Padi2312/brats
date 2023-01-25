import nibabel as nib
import SimpleITK as sitk
import numpy as np
from torch import nn
import torch.nn.functional as F


def load_nifti_data(path: str):
    img = sitk.ReadImage(path, sitk.sitkFloat32)
    return sitk.GetArrayFromImage(img)


def get_image_slice(data, image_slice: int):
    return data[image_slice, :, :]
