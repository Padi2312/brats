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
from utils import load_nifti_data, get_image_slice


class BratsDataset2D(Dataset):
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
            seed = np.random.randint(2147836)
            random.seed(seed)
            image = self.custom_transforms(image)
            random.seed(seed)
            mask = self.custom_transforms(mask)
        return image, mask

    def __preprocess_image(self, image):
        def internal_processing(raw_img):
            processed_image = torch.from_numpy(raw_img)
            processed_image = torch.nn.functional.normalize(processed_image)
            processed_image = self.__reshape_slices(processed_image)
            processed_image = torchvision.transforms.Resize(
                size=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
            )(processed_image)
            processed_image = processed_image.to(torch.float)

            mean = torch.mean(processed_image)
            std = torch.std(processed_image)

            # Normalize each channel
            processed_image = (processed_image - mean) / std

            return get_image_slice(processed_image, 59)

        if image.shape[0] > 1:
            target_images = torch.zeros((4, config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
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
        new_mask = torchvision.transforms.Resize(
            size=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
        )(new_mask)
        new_mask = get_image_slice(new_mask, 59)
        new_mask = torch.nn.functional.one_hot(new_mask.long(), 4)
        new_mask = torch.movedim(new_mask, -1, 0)
        new_mask = new_mask.to(torch.float)
        return new_mask

    def __reshape_slices(self, obj):
        return obj[14:142, ...]


if __name__ == "__main__":
    paths = Paths(config.DATASET_PATH)
    image_paths, mask_paths = paths.get_file_path_list_multi_channel()
    custom_transforms = transforms.Compose(
        transforms=[
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(90),
        ]
    )

    img_path, mask_path = image_paths[0:1], mask_paths[0:1]
    dataset = BratsDataset2D(
        image_paths=img_path,
        mask_paths=mask_path,
        custom_transforms=custom_transforms,
    )
    non_trans_dataset = BratsDataset2D(img_path, mask_path)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    non_trans_data_loader = DataLoader(non_trans_dataset, batch_size=1, shuffle=False)

    x, y = next(iter(data_loader))
    non_tr_x, non_tr_y = next(iter(non_trans_data_loader))

    size_bytes_x = x.nelement() * x.element_size()
    size_bytes_y = y.nelement() * y.element_size()
    print(x.shape, y.shape)
    print(size_bytes_x // 1000000, "MB")
    print(size_bytes_y // 1000000, "MB")

    fig, ax = plt.subplots(ncols=3)
    ax[0].imshow(x[0][2], cmap="gray")
    ax[0].imshow(y[0][2], alpha=0.3, cmap="viridis")

    # ax[1].imshow(non_tr_x[0][2][59], cmap="gray")
    # ax[1].imshow(non_tr_y[0][59], alpha=0.3, cmap="viridis")

    # ax[2].imshow(non_tr_x[0][2][59], cmap="gray")
    # plt.imshow(x[0][2][59], cmap="gray")
    # plt.imshow(y[0][59], alpha=0.3, cmap="viridis")
    # plt.imshow(y[0][0][59], alpha=0.2, cmap="Blues")
    # plt.imshow(y[0][1][59], alpha=0.2, cmap="Greens")
    # plt.imshow(y[0][2][59], alpha=0.2, cmap="Reds")
    # plt.imshow(y[0][3][59], alpha=0.2, cmap="Purples")
    plt.show()
