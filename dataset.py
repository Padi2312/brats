import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms

import config
from paths import Paths
from utils import RandomChoice, load_nifti_data


class BratsDataset(Dataset):
    def __init__(self, image_paths, mask_paths, custom_transforms=None):
        self.custom_transforms = custom_transforms
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image = None
        mask = load_nifti_data(self.mask_paths[index])

        if self.image_paths.ndim == 1:
            image = load_nifti_data(self.image_paths[index])
            image = np.expand_dims(image, axis=0)
        else:
            image = np.zeros((4, 155, 240, 240))
            for i, path in enumerate(self.image_paths[index]):
                curr_image = load_nifti_data(path)
                image[i] = np.array(curr_image)

        image = self.__preprocess_image(image)
        mask = self.__preprocess_mask(mask)

        if self.custom_transforms is not None:
            state = torch.get_rng_state()
            image = self.custom_transforms(image)
            torch.set_rng_state(state)
            mask = self.custom_transforms(mask)

        return image, mask

    def __preprocess_image(self, image):
        def internal_processing(raw_img):
            processed_image = torch.from_numpy(raw_img)
            processed_image = torch.nn.functional.normalize(processed_image)
            processed_image = self.__reshape_slices(processed_image)
            processed_image = self.__resize_image(processed_image)
            processed_image = processed_image.to(torch.float)
            return processed_image

        if image.shape[0] > 1:
            target_images = torch.zeros(
                (4, 128, config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
            )
            for i, img in enumerate(image):
                target_images[i] = internal_processing(img)
            return target_images
        else:
            new_image = internal_processing(image[0])
            new_image = new_image.unsqueeze(0)
            return new_image.to(torch.float)

    def __preprocess_mask(self, mask):
        mask[mask == 4] = 3
        mask = self.__reshape_slices(mask)
        new_mask = torch.from_numpy(mask)
        new_mask = self.__resize_image(new_mask)
        # new_mask = torch.nn.functional.one_hot(new_mask.long(), num_classes=4)
        # new_mask = new_mask.to(torch.float)
        # new_mask = new_mask.movedim(-1, 0)
        # new_mask = new_mask.unsqueeze(0)
        return new_mask

    def __reshape_slices(self, obj):
        return obj[14:142, ...]

    def __resize_image(self, obj):
        return torchvision.transforms.Resize(
            size=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
        )(obj)


if __name__ == "__main__":
    paths = Paths(config.DATASET_PATH)
    image_paths, mask_paths = paths.get_file_path_list_multi_channel()
    custom_transforms = transforms.Compose(
        transforms=[
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
        ]
    )

    img_path, mask_path = image_paths[0:3], mask_paths[0:3]
    dataset = BratsDataset(
        image_paths=img_path,
        mask_paths=mask_path,
        custom_transforms=custom_transforms,
    )
    raw_dataset = BratsDataset(img_path, mask_path)
    x, y = next(iter(dataset))
    raw_x, raw_y = next(iter(raw_dataset))

    print(f"Image shape: {x.shape} | Mask shape: {y.shape}")

    fig, ax = plt.subplots(ncols=2, figsize=(10, 10))
    ax[0].imshow(x[2][59], cmap="gray")
    ax[0].imshow(y[59], alpha=0.3, cmap="viridis")

    ax[1].imshow(raw_x[2][59], cmap="gray")
    ax[1].imshow(raw_y[59], alpha=0.3, cmap="viridis")

    plt.show()
