import SimpleITK as sitk
import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms
import numpy as np
import config
import os
import matplotlib.pyplot as plt
from PIL import Image


def get_numpy_array(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path, sitk.sitkInt16))


name = "BraTS2021_00000"
t2_path = f"C:\\bratsdata\\{name}\\{name}_t2.nii.gz"
t1_path = f"C:\\bratsdata\\{name}\\{name}_t1.nii.gz"
t1ce_path = f"C:\\bratsdata\\{name}\\{name}_t1ce.nii.gz"
flair_path = f"C:\\bratsdata\\{name}\\{name}_flair.nii.gz"
mask_path = f"C:\\bratsdata\\{name}\\{name}_seg.nii.gz"
pickle_path = f"D:\\bratsdata_processed\\{name}\\{name}_80.npz"

z = 90
img = torch.from_numpy(get_numpy_array(t1_path))
img = img[z, :, :]
img = img.unsqueeze(0)
print(torch.max(img[0]))


def resize_image(img):
    return transforms.Compose(
        [
            # transforms.CenterCrop((128 + 40, 128 + 40)),
            # transforms.Resize((128, 128), transforms.InterpolationMode.NEAREST),
            transforms.RandomPosterize(1),
        ]
    )(img)


processed = img
processed = processed.to(torch.int8)
# processed = resize_image(img)
fig, ax = plt.subplots(ncols=2, figsize=(10, 10))
ax[0].imshow(img[0])
ax[1].imshow(processed[0])
plt.show()

# data = np.load(pickle_path)
# img = torch.from_numpy(data["label"])
# print(torch.max(img))
# plt.imshow(img)
# plt.show()
