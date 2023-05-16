import torch
import SimpleITK as sitk


def get_tensor_from_nifti(path):
    return torch.from_numpy(
        sitk.GetArrayFromImage(sitk.ReadImage(path, sitk.sitkInt16))
    )
